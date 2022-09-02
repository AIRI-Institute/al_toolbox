import json
import logging
import os
from math import ceil
from pathlib import Path
from typing import List, Union

import dill
import numpy as np
import torch
import torch.multiprocessing as mp
from torch import optim
from tqdm import tqdm

log = logging.getLogger()


def point_to_device(point, device):
    if isinstance(point, torch.Tensor):
        return point.to(device)
    elif isinstance(point, list):
        return [el.to(device) for el in point]
    elif isinstance(point[next(iter(point.keys()))], list):
        return {k: torch.stack(v, dim=1).to(device) for k, v in point.items()}
    return {k: v.to(device) for k, v in point.items()}


def calculate_grad(model, optimizer, point):

    optimizer.zero_grad()

    point = point_to_device(point, next(model.parameters()).device)
    point = {k: v for k, v in point.items() if k in model.forward.__code__.co_varnames}
    loss = model(**point)["loss"]

    loss.backward()
    loss_gradient = torch.cat(
        [param.grad.view(-1) for param in model.parameters() if param.grad is not None]
    )
    return loss_gradient


def calculate_grad_flair(model, optimizer, point):
    optimizer.zero_grad()

    loss = model.forward_loss(point)
    loss = loss[0]

    loss.backward()
    loss_gradient = torch.cat(
        [param.grad.view(-1) for param in model.parameters() if param.grad is not None]
    )
    return loss_gradient


def calculate_tracin_score_one_epoch(model, optimizer, train_point, test_point, nu=1):

    train_grad = calculate_grad(model, optimizer, train_point)
    test_grad = calculate_grad(model, optimizer, test_point)
    return nu * (train_grad @ test_grad)


def load_dataloader(dataloader_path):
    with open(dataloader_path, "rb") as f:
        dataloader = dill.load(f)
    return dataloader


def calculate_tracin_outlier_score_one_model(
    model_load_func,
    args: Union,  # args for model_load function
    weights_path,
    dataloader_path,
    work_dir,
    n_epoch,
    nu=1,
    cuda_device=0,
    framework="transformers",
):

    log.info("Loading model and dataloader")
    model = model_load_func(*args)
    if framework == "flair":
        import flair

        # bilstm-crf case
        # set device before creating model
        flair.device = torch.device(f"cuda:{cuda_device}")
        model.load(weights_path)
        model.to(torch.device(f"cuda:{cuda_device}"))
        # also manually move all embeddings to device
        for embedding in model.embeddings.embeddings:
            embedding.to(torch.device(f"cuda:{cuda_device}"))
    else:
        model.load_state_dict(torch.load(weights_path))
        model.cuda(cuda_device)

    dataloader = load_dataloader(dataloader_path)
    log.info("Done with loading model and dataloader")

    optimizer = optim.SGD(model.parameters(), lr=1)
    scores = []

    grad_func = calculate_grad_flair if framework == "flair" else calculate_grad
    log.info("Start calculating TracIn scores")
    for i, point in enumerate(tqdm(dataloader)):
        point_to_device(point, 0)

        point_grad = grad_func(model, optimizer, point)
        point_epoch_score = nu * (point_grad @ point_grad.T)

        scores.append(float(point_epoch_score.item()))
    log.info("Done with calculating TracIn scores")

    with open(Path(work_dir) / f"scores_epoch_{n_epoch}.json", "w") as f:
        json.dump(scores, f)
    return scores


def load_model(model_path: str or Path) -> torch.nn.Module:
    with open(model_path, "rb") as f:
        model = dill.load(f)
    return model


def load_flair_model(model_path, idx_to_tag, config):
    # here we create model, because we can't pickle it
    from ..al4nlp.models.bilstm_crf import create_flair_model_tokenizer

    cache_dir = Path(config.cache_dir) if config.cache_dir is not None else None
    model, tokenizer = create_flair_model_tokenizer(
        config.target_model,
        idx_to_tag,
        config.data.label_name,
        config.seed,
        cache_dir=cache_dir,
    )
    return model


def calculate_outlier_scores(
    model_path: str or Path,
    weights_paths: List[str or Path],
    dataloader_path: str or Path,
    work_dir: str or Path,
    max_num_processes: int = None,
    task: str = "ner",
    nu: float or int = 1,
    framework: str = "transformers",
    idx_to_tag: dict = None,
    config: dict = None,
):
    torch.cuda.empty_cache()
    mp.set_start_method("spawn", force=True)

    processes = []
    cuda_devices = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
    total_procs = len(weights_paths)
    num_cudas = len(cuda_devices)
    num_procs_per_cuda = min(max_num_processes, ceil(len(weights_paths) / num_cudas))
    num_procs_per_core = ceil(
        total_procs / (max_num_processes * num_cudas)
    )  # ~ num batches

    load_model_func = load_flair_model if framework == "flair" else load_model
    model_args = (
        [model_path, idx_to_tag, config] if framework == "flair" else [model_path]
    )
    for n_proc_per_core in range(num_procs_per_core):
        for n_proc_per_cuda in range(num_procs_per_cuda):
            for i_cuda, cuda_device in enumerate(cuda_devices):
                n_epoch = (
                    i_cuda
                    + n_proc_per_cuda * num_cudas
                    + n_proc_per_core * (num_cudas * num_procs_per_cuda)
                )
                if n_epoch >= total_procs:
                    break
                log.info(
                    f"Starting process for epoch {n_epoch} on cuda device {cuda_device}, a process number "
                    f"{n_proc_per_cuda} on this cuda device."
                )
                p = mp.Process(
                    target=calculate_tracin_outlier_score_one_model,
                    args=(
                        load_model_func,
                        model_args,
                        weights_paths[n_epoch],
                        dataloader_path,
                        work_dir,
                        n_epoch,
                        nu,
                        i_cuda,
                        framework,
                    ),
                )
                p.start()
                processes.append(p)
                torch.cuda.empty_cache()
        for p in processes:
            p.join()

    log.info(f"TracIn scores calculation done.")
    scores = []
    for n_epoch in range(total_procs):
        with open(Path(work_dir) / f"scores_epoch_{n_epoch}.json") as f:
            scores.append(json.load(f))

    if task == "ner":
        dataloader = load_dataloader(dataloader_path)
        if framework == "flair":
            lengths = [len(point[0].tokens) for point in dataloader]
        else:
            lengths = [len(point["idx_first_bpe"][0]) for point in dataloader]
        final_scores = np.mean(scores, axis=0) / lengths
        return final_scores, scores, lengths

    elif task == "cls":
        final_scores = np.mean(scores, axis=0)
        return final_scores, scores
    else:
        raise NotImplementedError

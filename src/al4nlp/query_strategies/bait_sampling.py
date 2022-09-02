import gc
import logging
from typing import Union

import numpy as np
import torch
from datasets.arrow_dataset import Dataset
from scipy import stats
from tqdm import tqdm

from .al_strategy_utils import take_idx
from ..utils.get_embeddings import get_exp_grad_embedding
from ..utils.transformers_dataset import TransformersDataset

log = logging.getLogger()


def bait_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    data_config=None,
    data_is_tokenized=False,
    batch_size=100,
    to_numpy=True,
    fishIdentity=0,
    fishInit=1,
    lamb=1,
    backwardSteps=1,
    **bait_kwargs,
):
    """
    Selects batches of samples by optimizing a bound on the MLE error in terms of the Fisher information.
    https://arxiv.org/abs/2106.09675
    """
    data_config = data_config if data_config is not None else model.data_config
    kwargs = dict(
        # General
        model=model.model,
        batch_size=batch_size,
        to_numpy=False,
        data_is_tokenized=data_is_tokenized,
        tokenizer=model.tokenizer,
        task=model.task,
        text_name=data_config["text_name"],
        label_name=data_config["label_name"],
    )

    xt_train = get_exp_grad_embedding(dataloader_or_data=X_train, **kwargs)
    xt_test = get_exp_grad_embedding(dataloader_or_data=X_pool, **kwargs)
    xt = torch.cat((xt_train, xt_test))
    xt2 = xt_train

    # get fisher
    if fishIdentity == 0:
        print("getting fisher matrix ...", flush=True)
        batchSize = 100
        nClass = model.num_labels
        fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
        rounds = int(np.ceil(len(xt) / batchSize))
        for i in range(int(np.ceil(len(xt) / batchSize))):
            xt_ = xt[i * batchSize : (i + 1) * batchSize].cuda()
            op = (
                torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / (len(xt)), 0)
                .detach()
                .cpu()
            )
            fisher = fisher + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()
    else:
        fisher = torch.eye(xt.shape[-1])

    # get fisher only for samples that have been seen before
    batchSize = 100

    init = torch.zeros(xt.shape[-1], xt.shape[-1])
    rounds = int(np.ceil(len(xt2) / batchSize))
    if fishInit == 1:
        for i in range(int(np.ceil(len(xt2) / batchSize))):
            xt_ = xt2[i * batchSize : (i + 1) * batchSize].cuda()
            op = (
                torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / (len(xt2)), 0)
                .detach()
                .cpu()
            )
            init = init + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

    query_idx = select(
        xt_test,
        n_instances,
        fisher,
        init,
        lamb=lamb,
        backwardSteps=backwardSteps,
        nLabeled=len(X_train),
    )

    query = take_idx(X_pool, query_idx)

    # Uncertainty estimates for BAIT are not defined
    uncertainty_estimates = np.zeros(len(X_pool))

    return query_idx, query, uncertainty_estimates


def select(X, K, fisher, iterates, lamb=1, backwardSteps=0, nLabeled=0):
    numEmbs = len(X)
    indsAll = []
    dim = X.shape[-1]
    rank = X.shape[-2]

    currentInv = torch.inverse(
        lamb * torch.eye(dim).cuda() + iterates.cuda() * nLabeled / (nLabeled + K)
    )
    X = X * np.sqrt(K / (nLabeled + K))
    fisher = fisher.cuda()

    # forward selection
    for i in tqdm(range(int((backwardSteps + 1) * K))):

        xt_ = X.cuda()
        innerInv = torch.inverse(
            torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)
        ).detach()
        innerInv[torch.where(torch.isinf(innerInv))] = (
            torch.sign(innerInv[torch.where(torch.isinf(innerInv))])
            * np.finfo("float32").max
        )
        traceEst = torch.diagonal(
            xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv,
            dim1=-2,
            dim2=-1,
        ).sum(-1)

        xt = xt_.cpu()
        del xt, innerInv
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        traceEst = traceEst.detach().cpu().numpy()

        dist = traceEst - np.min(traceEst) + 1e-10
        dist = dist / np.sum(dist)
        sampler = stats.rv_discrete(values=(np.arange(len(dist)), dist))
        ind = sampler.rvs(size=1)[0]
        for j in np.argsort(dist)[::-1]:
            if j not in indsAll:
                ind = j
                break

        indsAll.append(ind)

        xt_ = X[ind].unsqueeze(0).cuda()
        innerInv = torch.inverse(
            torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)
        ).detach()
        currentInv = (
            currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv
        ).detach()[0]

    # backward pruning
    for i in tqdm((range(len(indsAll) - K))):
        # select index for removal
        xt_ = X[indsAll].cuda()
        innerInv = torch.inverse(
            -1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)
        ).detach()
        traceEst = torch.diagonal(
            xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv,
            dim1=-2,
            dim2=-1,
        ).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()

        # compute new inverse
        xt_ = X[indsAll[delInd]].unsqueeze(0).cuda()
        innerInv = torch.inverse(
            -1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)
        ).detach()
        currentInv = (
            currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv
        ).detach()[0]

        del indsAll[delInd]

    del xt_, innerInv, currentInv
    torch.cuda.empty_cache()
    gc.collect()
    return np.array(indsAll)

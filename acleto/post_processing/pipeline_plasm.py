import logging
import os
from pathlib import Path
from shutil import rmtree
from typing import Union, Tuple, List

import numpy as np
import transformers.utils.logging
from datasets import Dataset, load_dataset, concatenate_datasets
from hydra import compose, initialize, core
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .calculate_tracin_score import calculate_outlier_scores
from .plasm import label_data, concatenate_data
from ..al4nlp.constructors.construct_wrapper import construct_wrapper
from ..al4nlp.pool_subsampling_strategies.random_subsampling import random_subsampling
from ..al4nlp.utils.general import dill_dump, get_target_model_checkpoints, add_new_model_to_time_dict, json_load
from ..al4nlp.utils.transformers_dataset import TransformersDataset

log = logging.getLogger(__name__)
transformers.utils.logging.disable_progress_bar()


OmegaConf.register_new_resolver(
    "to_string", lambda x: x.replace("/", "_").replace("-", "_"), replace=True
)
OmegaConf.register_new_resolver(
    "get_patience_value", lambda dev_size: 1000 if dev_size == 0 else 5, replace=True
)
DEFAULT_CONFIG_NAME = "cls_plasm"

def pipeline_plasm(
        data_dir: Union[str, Path],
        annotation_or_annotation_dir: Union[str, Path, List[List[int]], np.ndarray],
        config_name: str = None,
        train_model: bool = True,

) -> Tuple[Union[Dataset, TransformersDataset], "ModelWrapper"]:
    # Initialize config
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../jupyterlab_demo/configs/")
    config = compose(config_name=config_name or DEFAULT_CONFIG_NAME)
    # Load data
    log.info("Starting post processing")
    data_dir = Path(data_dir)
    labeled_data = load_dataset('json', data_files=str(data_dir / 'labeled.json'), split="train")
    unlabeled_data = load_dataset('json', data_files=str(data_dir / 'unlabeled.json'), split="train")
    # Load annotation
    if isinstance(annotation_or_annotation_dir, (list, np.ndarray)):
        annotation = annotation_or_annotation_dir
    else:
        annotation = json_load(Path(annotation_or_annotation_dir) / "annotation.json")
    # Load label2id
    tags = json_load(data_dir / "tags.json")
    id2label = {idx: tag for idx, tag in enumerate(tags)}
    label2id = {v: k for k, v in id2label.items()}
    labeled_idx = [i for i, inst_ann in enumerate(annotation) if inst_ann is not None]
    # Remove Nones and transform labels to ids
    annotation = [list(map(lambda tag: label2id[tag], annotation[idx])) for idx in labeled_idx]
    annotated_data = unlabeled_data.select(labeled_idx).add_column(config.data.label_name, annotation)
    labeled_data = concatenate_datasets([labeled_data, annotated_data], info=labeled_data.info)
    ### Specify pseudo-labeled data peculiarities
    # Whether to use label_smoothing for pseudo labeled data;
    # True -> use classic LS, 'natural' -> use returned probas, None -> do not use
    pl_label_smoothing = config.post_processing.get("label_smoothing")
    # If we want to increase the weight of the "real labeled data" compared to pseudo labeled data
    labeled_weight = config.post_processing.get("labeled_weight", 1)
    # Uncertainty threshold
    unc_threshold = config.post_processing.get("uncertainty_threshold", None)
    # We can either filter by absolute value or by a quantile
    unc_filt_by_quant = unc_threshold == "adaptive" or config.post_processing.get("filter_by_quantile", None)
    # Whether pseudo-labeling will be performed over a subsample of the data
    use_subsample_for_pl = config.post_processing.get("use_subsample_for_pl", False)
    # Whether to use TracIn
    use_tracin = config.post_processing.tracin.get("use", False)
    # Whether to use TracIn
    tracin_quantile = config.post_processing.tracin.get("quantile", "adaptive")
    log.info(
        f"Pseudo-labeling:\n" +
        f"PL label smoothing factor: {pl_label_smoothing}\n" +
        f"Labeled data weight: {labeled_weight}\n" +
        f"Using TracIn: {use_tracin}\n" +
        (f"TracIn quantile: {tracin_quantile}" if use_tracin else "") +
        f"Uncertainty threshold: {unc_threshold}\n" +
        (f"Uncertainty filter by quantile: {unc_filt_by_quant}\n" if unc_threshold is not None else "") +
        f"PL using subsample: {use_subsample_for_pl}\n"
    )
    # Construct and train a pseudo-labeling model
    labeling_wrapper = construct_wrapper(
        config,
        model_cfg=config.labeling_model,
        dev_data=None,
        labels_or_id2label=id2label,
        name="labeling"
    )
    log.info("Labeling wrapper constructed. Started fitting...")
    labeling_wrapper.fit(labeled_data, from_scratch=True)
    log.info("Labeling wrapper fitted.")
    # Setting adaptive parameters if necessary
    lab_model_quality = getattr(labeling_wrapper, "best_metric", None)
    if lab_model_quality is None:
        if tracin_quantile == "adaptive":
            log.warning(
                "You set TracIn filtering share to be adaptive but no dev data was used for evaluation. " +
                "Using value 0.1."
            )
            tracin_quantile = 0.1
        if pl_label_smoothing == "adaptive":
            log.warning(
                "You set PL label smoothing coef. to be adaptive but no dev data was used for evaluation. " +
                "Disabling PL label smoothing."
            )
            pl_label_smoothing = None
        if unc_threshold == "adaptive":
            log.warning(
                "You set uncertainty threshold coef. to be adaptive but no dev data was used for evaluation. " +
                "Disabling uncertainty threshold filtering."
            )
            unc_threshold = None

    # Set adaptive coefs
    if pl_label_smoothing == "adaptive":
        pl_label_smoothing = lab_model_quality
        log.info(f"Using pseudo-labeling parameter {pl_label_smoothing:.5f}.")
    if unc_threshold == "adaptive":
        if unc_filt_by_quant:
            unc_threshold = 1 - lab_model_quality
        else:
            unc_threshold = lab_model_quality
        log.info(f"Using uncertainty parameter {unc_threshold:.5f}.")
    if tracin_quantile == "adaptive":
        tracin_quantile = 1 - lab_model_quality
        log.info(f"Using TracIn filtering coef {tracin_quantile:.5f}.")
    # Set data for pseudo-labeling
    if use_subsample_for_pl:
        if isinstance(use_subsample_for_pl, float):
            use_subsample_for_pl = round(len(unlabeled_data) * use_subsample_for_pl)
        subsample_idx = random_subsampling(
            np.ones(len(unlabeled_data)), use_subsample_for_pl
        )
        unlabeled_data = unlabeled_data.select(subsample_idx)
    # Pseudo labeling!
    log.info("Starting pseudo-labeling...")
    pseudo_labeled_data, filtered_data_share = label_data(
        unlabeled_data,
        labeling_wrapper,
        threshold=unc_threshold,
        uncertainty_filter_by_quantile=unc_filt_by_quant,
        pl_label_smoothing=pl_label_smoothing,
        id2label=id2label
    )
    log.info("Pseudo-labeling done.")
    # Concatenate the labeled-by-oracle and pseudo-labeled data
    successor_train_data = concatenate_data(
        labeled_data, pseudo_labeled_data, labeled_weight
    )
    if not train_model:
        return successor_train_data

    # Train the successor model
    successor_wrapper = construct_wrapper(
        config,
        model_cfg=config.successor_model,
        dev_data=None,
        labels_or_id2label=id2label,
        name="successor"
    )
    log.info("Successor wrapper constructed. Started fitting...")
    successor_wrapper.fit(successor_train_data)
    log.info("Successor wrapper fitted.")
    tracin_quantile -= filtered_data_share
    if not use_tracin or tracin_quantile <= 0:
        return successor_train_data, successor_wrapper

    tokenizer = successor_wrapper.tokenizer
    tokenized_data = successor_wrapper.tokenize_data(
        tokenizer=tokenizer,
        data=pseudo_labeled_data,
        text_name=successor_wrapper.data_config["text_name"],
        label_name=successor_wrapper.data_config["label_name"],
    )

    framework = config.framework.name
    framework_is_flair = framework == "flair"
    if framework_is_flair:
        from flair.datasets import DataLoader as FlairDataLoader

        dataloader = FlairDataLoader(tokenized_data, batch_size=1, )
    else:
        dataloader = DataLoader(
            tokenized_data,
            batch_size=1,
            pin_memory=True,
        )

    tracin_path = data_dir / "tracin"
    tracin_path.mkdir(exist_ok=True)
    model_path = tracin_path / f"tmp_successor_model_{config.seed}"
    model_weights_paths = get_target_model_checkpoints(config, framework)
    dataloader_path = (tracin_path / "tmp_dataloader")

    if framework_is_flair:
        # flair models don't work with dill
        successor_wrapper.model.save(model_path)
    else:
        dill_dump(successor_wrapper.model, model_path)

    dill_dump(dataloader, dataloader_path)
    # Calculate TracIn score
    idx_to_tag = (
        successor_wrapper.data_config["idx_to_tag"] if framework_is_flair else None
    )
    log.info("TracIn started")
    scores, *_ = calculate_outlier_scores(
        model_path=model_path,
        weights_paths=model_weights_paths,
        dataloader_path=dataloader_path,
        work_dir=tracin_path,
        max_num_processes=config.post_processing.tracin.max_num_processes,
        task=successor_wrapper.task,
        nu=config.post_processing.tracin.nu,
        framework=framework,
        idx_to_tag=idx_to_tag,
        config=config,
    )
    log.info("TracIn finished")
    # Change fp16 dict
    successor_wrapper._trainer_kwargs["fp16"] = successor_wrapper._trainer_kwargs.get(
        "final_model_fp16", False
    )
    # Modify the name to avoid overwriting time-file
    successor_wrapper.name = successor_wrapper.name + f"_tracin_quantile_{np.round(tracin_quantile, 4)}"
    # Add initial empty fit/predict lists of the model to the time dict
    add_new_model_to_time_dict(successor_wrapper.time_dict_path, successor_wrapper.name)
    # Load key metrics of the successor model (f1-score for NER, accuracy for CLS)
    outliers = np.argwhere(scores > np.quantile(scores, (1 - tracin_quantile))).ravel()
    pseudo_labeled_data_to_use = [
        pseudo_labeled_data[i]
        for i in range(len(pseudo_labeled_data))
        if i not in outliers
    ]

    if successor_wrapper.task == "ner":
        tokenization_kwargs = {"is_split_into_words": True}
    elif successor_wrapper.task == "cls":
        tokenization_kwargs = {}
    else:
        raise NotImplementedError

    pseudo_labeled_data_to_use = TransformersDataset(
        pseudo_labeled_data_to_use,
        text_column_name=config.data.text_name,
        label_column_name=config.data.label_name,
        tokenization_kwargs=tokenization_kwargs,
        task=pseudo_labeled_data.task,
        id2label=pseudo_labeled_data.id2label,
        label_smoothing=pseudo_labeled_data.label_smoothing,
    )
    labeled_weight = successor_wrapper.model_config.training.get("labeled_weight", 1.0)
    filtered_data = concatenate_data(
        labeled_data, pseudo_labeled_data_to_use, labeled_weight
    )
    log.info("Successor fitting...")
    successor_wrapper.fit(filtered_data)
    log.info("Successor fitted.")
    rmtree(successor_wrapper._trainer_kwargs["serialization_dir"])
    os.remove(model_path)
    os.remove(dataloader_path)
    return filtered_data, successor_wrapper

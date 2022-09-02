import logging
from functools import partial
from pathlib import Path
from typing import Union, Tuple

import numpy as np
from datasets.arrow_dataset import Dataset as ArrowDataset
from omegaconf.omegaconf import ListConfig

from ..al4nlp.utils.general import json_load
from ..al4nlp.utils.transformers_dataset import TransformersDataset

log = logging.getLogger()


def label_data(
    data,
    model,
    threshold: Union[None, float, str] = None,
    uncertainty_filter_by_quantile: bool = False,
    pl_label_smoothing: Union[bool, float, str] = False,
    **kwargs
) -> Tuple[TransformersDataset, float]:
    use_threshold = (threshold is not None) and (
        isinstance(threshold, (ListConfig, list)) or (threshold > 0)
    )
    task = model.task
    filtered_data_share = 0.0
    dataset_kwargs = {}

    log.info("Starting pseudo-labeling the unlabeled data.")
    idx_to_save = []
    if task == "cls":
        probas = model.predict_proba(data, remove_padding=True)
        if use_threshold:
            max_prob = np.max(probas, axis=-1)
            if uncertainty_filter_by_quantile:
                num_instances_to_filter = round(len(max_prob) * threshold)
                idx_to_save = np.argsort(max_prob)[num_instances_to_filter:]
                filtered_data_share = threshold
            else:
                idx_to_save = np.argwhere(max_prob > threshold).ravel()
                filtered_data_share = 1 - len(idx_to_save) / len(probas)
            log.info(
                f"With uncertainty parameter {threshold}, {len(idx_to_save)} instances" +
                f"will be kept ({round(1 - filtered_data_share, 4) * 100}% of data)."
            )
        if isinstance(pl_label_smoothing, float):
            wrong_class_proba = (1 - pl_label_smoothing) / (probas.shape[-1] - 1)
            pseudo_labels = (
                np.zeros(probas.shape, dtype=probas.dtype) + wrong_class_proba
            )
            pseudo_labels[
                range(len(probas)), np.argmax(probas, axis=-1)
            ] = pl_label_smoothing
        elif pl_label_smoothing == "natural":
            pseudo_labels = probas
        elif not pl_label_smoothing:
            pseudo_labels = np.argmax(probas, axis=-1)
        else:
            raise NotImplementedError
        dataset_kwargs["label_smoothing"] = pl_label_smoothing != False
    elif task == "ner":
        probas = model.predict_proba(data, remove_padding=True)
        if use_threshold:
            scores = 1 - np.array(
                [-np.sum(np.log(np.max(i, axis=1))) / len(i) for i in probas]
            )
            if uncertainty_filter_by_quantile:
                num_instances_to_filter = round(len(scores) * threshold)
                idx_to_save = np.argsort(scores)[num_instances_to_filter:]
                filtered_data_share = threshold
            else:
                idx_to_save = np.argwhere(scores > threshold).ravel()
                filtered_data_share = 1 - len(idx_to_save) / len(probas)
            log.info(
                f"With uncertainty parameter {threshold}, {len(idx_to_save)} instances" +
                f"will be kept ({(1 - filtered_data_share) * 100:.5f}% of data)."
            )
        pseudo_labels = np.array(
            [np.argmax(inst_logits, axis=1) for inst_logits in probas]
        )
    else:
        raise NotImplementedError
    label_column_name = model.data_config["label_name"]

    if task == "ner":
        if isinstance(data, ArrowDataset):
            if label_column_name in data.features:
                dataset_kwargs["id2label"] = {
                    i: tag
                    for i, tag in enumerate(data.features[label_column_name].feature.names)
                }
            else:
                dataset_kwargs["id2label"] = kwargs.get("id2label")
        elif isinstance(data, TransformersDataset):
            dataset_kwargs["id2label"] = data.id2label

    if use_threshold:
        new_instances = list(data.select(idx_to_save))
        pseudo_labels = pseudo_labels[idx_to_save]
    else:
        new_instances = list(data)

    for i, inst in enumerate(new_instances):
        new_instances[i][label_column_name] = pseudo_labels[i]
    new_data = TransformersDataset(new_instances, task=task, **dataset_kwargs)

    return new_data, filtered_data_share


def concatenate_data(
    labeled_data: Union,
    pseudo_labeled_data: TransformersDataset,
    labeled_weight: Union[float, int] = 1.0,
):
    # Check whether label smoothing is applied
    if pseudo_labeled_data.label_smoothing:
        label_name = pseudo_labeled_data.label_column_name
        labels_vector_length = len(pseudo_labeled_data[label_name][0])
        labels_to_ohe_func = partial(
            _transform_labels_to_ohe,
            labels_vector_length=labels_vector_length,
            label_name=label_name,
        )
        labeled_data = labeled_data.map(labels_to_ohe_func)

    # For TracIn part

    all_data = pseudo_labeled_data.add(labeled_data, inplace=False)
    if labeled_weight != 1.0:
        weight = [1 for _ in range(len(pseudo_labeled_data))] + [
            labeled_weight for _ in range(len(labeled_data))
        ]
        all_data.add_column("weight", weight)

    return all_data


def _transform_labels_to_ohe(instance, labels_vector_length=None, label_name="label"):
    label = instance[label_name]
    ohe_label = np.zeros(labels_vector_length)
    ohe_label[label] = 1
    return {label_name: ohe_label}


def _get_pseudo_labeling_model_metric_value(work_dir, task):
    metrics_dict = json_load(Path(work_dir) / f"successor_metrics.json")
    metric = "overall_f1" if task == "ner" else "accuracy"
    key_metric_value = metrics_dict[f"test_{metric}"][-1]
    return key_metric_value

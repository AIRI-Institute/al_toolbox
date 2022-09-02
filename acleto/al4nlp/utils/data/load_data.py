from typing import Union, Tuple

from datasets import Dataset

from .load_arbitrary_dataset import (
    load_arbitrary_dataset_for_cls,
    load_conll_format_dataset_for_ner,
    load_from_csv,
)
from .load_from_url import load_data_from_url
from .load_huggingface_dataset import load_huggingface_dataset
from ..transformers_dataset import TransformersDataset


def load_data(
    config, task, framework="transformers", cache_dir=None
) -> Tuple[
    Union[Dataset, TransformersDataset],
    Union[Dataset, TransformersDataset],
    Union[Dataset, TransformersDataset],
    Union[None, dict],
]:
    """

    :param config:
    :param task:
    :param framework:
    :param cache_dir:
    :return: train_dataset, dev_dataset, test_dataset, id2label
    """
    if config.path == "url":
        return load_data_from_url(config, cache_dir)
    elif config.path != "datasets":
        if task == "ner":
            return load_conll_format_dataset_for_ner(config)
        elif config.get("from_csv", None) is not None:
            return load_from_csv(config, task, cache_dir)
        else:
            return load_arbitrary_dataset_for_cls(config)
    return load_huggingface_dataset(config, task, cache_dir)

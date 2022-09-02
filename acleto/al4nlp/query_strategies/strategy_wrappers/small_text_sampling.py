from typing import Union

import numpy as np
from datasets import concatenate_datasets
from datasets.arrow_dataset import Dataset

from ..al_strategy_utils import take_idx
from ...utils.transformers_dataset import TransformersDataset


def small_text_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    small_text_strategy, 
    **kwargs,
):
    all_data = concatenate_datasets([X_train, X_pool])
    indices_labeled = np.arange(len(X_train))
    indices_unlabeled = np.arange(len(X_train), len(X_pool))
    small_text_strategy = small_text_strategy()
    
    query_idx = small_text_strategy.query(
        model, 
        all_data,
        indices_unlabeled, 
        indices_labeled,
        None,
        n=n_instances,
    )
    
    uncertainty_estimates = np.zeros(len(X_pool))
    query = take_idx(all_data, query_idx)

    return query_idx, query, uncertainty_estimates

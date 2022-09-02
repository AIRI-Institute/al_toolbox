from typing import Union

import numpy as np
from datasets.arrow_dataset import Dataset

from .al_strategy_utils import take_idx
from ..utils.transformers_dataset import TransformersDataset


def entropy_sampling(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **kwargs,
):
    """
    Selects instances with the largest prediction entropy.
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.28.9963&rep=rep1&type=pdf
    """
    probas = model.predict_proba(X_pool)
    uncertainty_estimates = np.sum(-probas * np.log(probas), axis=1)
    argsort = np.argsort(-uncertainty_estimates)
    query_idx = argsort[:n_instances]
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates

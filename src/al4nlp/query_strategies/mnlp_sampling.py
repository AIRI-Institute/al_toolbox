from typing import Union

import numpy as np
from datasets.arrow_dataset import Dataset

from .al_strategy_utils import (
    take_idx,
    calculate_mnlp_score,
    calculate_mnlp_scores_empty_penalty,
    get_query_idx_for_selecting_by_number_of_tokens,
)
from ..utils.transformers_dataset import TransformersDataset


def mnlp_sampling(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    select_by_number_of_tokens=True,
    **kwargs,
):
    """
    Normalize the likelihood over example lengths. https://arxiv.org/abs/1707.05928
    """
    probas = model.predict_proba(X_pool, remove_padding=True, test_mode=True)
    np.array([-np.sum(np.log(np.max(i, axis=1))) / len(i) for i in probas])

    empty_penalty = kwargs.get("empty_penalty", False)
    uncertainty_estimates = (
        calculate_mnlp_score(probas)
        if not empty_penalty
        else calculate_mnlp_scores_empty_penalty(probas)
    )
    argsort = np.argsort(-uncertainty_estimates)

    if select_by_number_of_tokens:
        query_idx = get_query_idx_for_selecting_by_number_of_tokens(
            X_pool, argsort, n_instances
        )
    else:
        query_idx = argsort[:n_instances]

    query = take_idx(X_pool, query_idx)
    return query_idx, query, uncertainty_estimates

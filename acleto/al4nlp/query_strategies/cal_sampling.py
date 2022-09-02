import logging
from typing import Union

import numpy as np
from datasets.arrow_dataset import Dataset

from .al_strategy_utils import (
    take_idx,
    calculate_cal_scores,
)
from ..utils.transformers_dataset import TransformersDataset

log = logging.getLogger()


def cal_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    num_nei=10,
    **cal_kwargs,
):
    """
    Selects unlabeled data points from the pool,
    whose predictive likelihoods diverge the most from their neighbors in the training set.
    https://arxiv.org/pdf/2109.03764.pdf
    """
    logits = model.predict_logits(X_pool)
    train_logits = model.predict_logits(X_train)
    kwargs = dict(
        # Necessary
        model_wrapper=model,
        data_train=X_train,
        data_test=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        to_numpy=True,
        probas=logits,
        train_probas=train_logits,
        num_nei=num_nei,
    )

    uncertainty_estimates = calculate_cal_scores(**kwargs)
    query_idx = np.argpartition(uncertainty_estimates, -n_instances)[-n_instances:]

    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates

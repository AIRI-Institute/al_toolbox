import logging
from typing import Union

import numpy as np
from datasets.arrow_dataset import Dataset

from .al_strategy_utils import (
    take_idx,
    choose_cm_samples,
)
from ..utils.transformers_dataset import TransformersDataset

log = logging.getLogger()


def cluster_margin_sampling(
    model, X_pool: Union[Dataset, TransformersDataset], n_instances: int, **kwargs
):
    """
    Select a diverse set of examples on which the model is least confident. The confidence of a model
    on an example is a difference between the largest two predicted class probabilities.In order to ensure diversity
    among the least confident examples, they are clustered using HAC with average-linkage.
    https://arxiv.org/pdf/2107.14263.pdf
    """
    instances_multiplier = kwargs.get("instances_multiplier", 1.25)
    random_query = kwargs.get("random_query", True)
    probas = model.predict_proba(X_pool)
    # reuse code from margin sampling
    # To get second max probas, need to sort the array since `.sort` modifies the array
    probas.sort(axis=1)
    max_probas = probas[:, -1]
    second_max_probas = probas[:, -2]
    uncertainty_estimates = 1 + second_max_probas - max_probas
    argsort = np.argsort(-uncertainty_estimates)
    # we have to choose n_instances with round-robin algorithm, so for this we
    # firstly choose 2 * n_instances
    query_idx = argsort[: int(instances_multiplier * n_instances)]
    cluster_labels = np.array(X_pool.clusters)[query_idx]
    # X_pool either transformer dataset or np array
    query_idx, samples_idx = choose_cm_samples(
        cluster_labels, query_idx, n_instances, random_query
    )
    query = take_idx(X_pool, query_idx)
    return query_idx, query, uncertainty_estimates

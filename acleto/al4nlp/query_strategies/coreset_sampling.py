import logging
from typing import Union

import numpy as np
from datasets.arrow_dataset import Dataset
from sklearn.metrics import pairwise_distances

from .al_strategy_utils import take_idx
from ..utils.get_embeddings import get_embeddings
from ..utils.transformers_dataset import TransformersDataset

log = logging.getLogger()


def furthest_first(X, X_set, n):
    m = np.shape(X)[0]
    if np.shape(X_set)[0] == 0:
        min_dist = np.tile(float("inf"), m)
    else:
        dist_ctr = pairwise_distances(X, X_set)
        min_dist = np.amin(dist_ctr, axis=1)

    idxs = []

    for i in range(n):
        idx = min_dist.argmax()
        idxs.append(idx)
        dist_new_ctr = pairwise_distances(X, X[[idx], :])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

    return np.array(idxs)


def coreset_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **coreset_kwargs,
):
    kwargs = dict(
        # General
        prepare_model=True,
        # use_activation=use_activation,
        # use_spectralnorm=use_spectralnorm,
        data_is_tokenized=False,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        to_numpy=True,
        # Tokenization
        tokenizer=model.tokenizer,
        task=model.task,
        text_name=model.data_config["text_name"],
        label_name=model.data_config["label_name"],
    )

    train_features = get_embeddings(model.model, X_train, **kwargs)
    test_features = get_embeddings(model.model, X_pool, **kwargs)

    query_idx = furthest_first(test_features, train_features, n_instances)

    query = take_idx(X_pool, query_idx)

    # Uncertainty estimates are not defined for CoreSet
    uncertainty_estimates = np.zeros(len(X_pool))

    return query_idx, query, uncertainty_estimates

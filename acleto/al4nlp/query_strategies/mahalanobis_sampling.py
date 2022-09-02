import logging
from typing import Union

import numpy as np
from datasets.arrow_dataset import Dataset

from .al_strategy_utils import (
    take_idx,
    calculate_mahalanobis_distance,
)
from ..utils.transformers_dataset import TransformersDataset

log = logging.getLogger()


def mahalanobis_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **mahalanobis_kwargs,
):
    batched = mahalanobis_kwargs.get("mahalanobis_batched", False)
    substrategy = mahalanobis_kwargs.get("mahalanobis_substrategy", "lc")
    use_class_probas = mahalanobis_kwargs.get("mahalanobis_use_class_probas", False)
    kwargs = dict(
        # Necessary
        model_wrapper=model,
        train_data=X_train,
        unlabeled_data=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        to_numpy=True,
        # Mahalanobis
        classwise=True,  # use an own centroid for each class
        batched=batched,
        use_v2=mahalanobis_kwargs.get("mahalanobis_use_v2", False),
        use_activation=mahalanobis_kwargs.get("mahalanobis_use_activation", False),
        use_spectralnorm=mahalanobis_kwargs.get("mahalanobis_use_spectralnorm", True),
    )
    if not batched:
        dists = calculate_mahalanobis_distance(**kwargs)
        if use_class_probas:
            log.info("Using probas for mahalanobis")
            label_name = model.data_config["label_name"]
            class_probas = np.bincount(
                X_train[label_name], minlength=model.num_labels
            ) / len(X_train)
            dists = dists * class_probas
        else:
            log.info("Not using probas for mahalanobis))")
        if substrategy == "lc":
            uncertainty_estimates = np.min(dists, axis=-1)
        elif substrategy == "margin":
            dists.sort(axis=1)
            max_dists = dists[:, 0]
            second_max_dists = dists[:, 1]
            uncertainty_estimates = (
                max_dists - second_max_dists
            )  # the greater the value, the more uncertain we are
        else:
            raise NotImplementedError
        argsort = np.argpartition(-uncertainty_estimates, kth=n_instances)
        query_idx = argsort[:n_instances]
    else:
        query_idx, uncertainty_estimates = calculate_mahalanobis_distance(
            n_instances=n_instances, **kwargs
        )
    query = take_idx(X_pool, query_idx)
    return query_idx, query, uncertainty_estimates

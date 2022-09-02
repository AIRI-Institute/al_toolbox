from typing import Union

from datasets.arrow_dataset import Dataset

from .al_strategy_utils import (
    take_idx,
    calculate_actune_scores,
)
from ..utils.transformers_dataset import TransformersDataset


def actune_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    query_strategy,
    **actune_kwargs,
):
    _, _, uncertainty_estimates = query_strategy(
        model, X_pool, n_instances, **actune_kwargs
    )

    probas = model.predict_proba(X_pool)
    kwargs = dict(
        # Necessary
        model=model,
        data_test=X_pool,
        # General
        data_is_tokenized=False,
        data_config=model.data_config,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        probas=probas,
        n_instances=n_instances,
        uncertainty_estimates=uncertainty_estimates,
        k=10,
    )

    query_idx, uncertainty_estimates = calculate_actune_scores(**kwargs)
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates

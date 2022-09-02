from typing import Union

import numpy as np
from datasets.arrow_dataset import Dataset

from .al_strategy_utils import (
    take_idx,
    _best_versus_second_best,
)
from ..utils.transformers_dataset import TransformersDataset


def breaking_ties_sampling(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **kwargs,
):
    """
    Selects instances which have a small margin between their most likely and second
    most likely predicted class.
    https://www.researchgate.net/publication/220321064_Active_Learning_to_Recognize_Multiple_Types_of_Plankton
    """
    probas = model.predict_proba(X_pool)
    uncertainty_estimates = np.apply_along_axis(
        lambda x: _best_versus_second_best(x), 1, probas
    )

    argsort = np.argsort(uncertainty_estimates)
    query_idx = argsort[:n_instances]
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates

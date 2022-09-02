import logging
from typing import Union

import numpy as np
import torch
from datasets.arrow_dataset import Dataset

from .al_strategy_utils import take_idx
from .strategy_utils.batchbald.batchbald import get_batchbald_batch
from .strategy_utils.batchbald.consistent_dropout import make_dropouts_consistent
from ..utils.transformers_dataset import TransformersDataset

log = logging.getLogger()


def batchbald_sampling(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **bald_kwargs,
):
    """
    Jointly score points by estimating the mutual information between a joint
    of multiple data points and the model parameters. https://arxiv.org/abs/1906.08158
    """
    mc_iterations = bald_kwargs.get("mc_iterations", 10)
    max_num_samples = bald_kwargs.get("max_num_samples", int(1e4))
    # requires enormous amount of memory
    device = "cpu"  # list(model.model.parameters())[0].device

    # Make dropout consistent inside huggingface model
    make_dropouts_consistent(model.model)

    if bald_kwargs.get("only_head_dropout", False):
        raise NotImplementedError
        # log_probas_N_K_C = model.predict_proba(X_pool, mc_dropout=True, mc_iterations=mc_iterations)
    else:
        probas = []
        for _ in range(mc_iterations):
            # Reset masks
            model.enable_dropout()
            model.disable_dropout()
            probas_iter = model.predict_proba(
                X_pool, use_predict_loop=True, to_numpy=False, to_eval_mode=False
            )
            probas.append(probas_iter)

        log_probas_N_K_C = torch.stack(probas, -2).log().to(device)

    query_idx, uncertainty_estimates = get_batchbald_batch(
        log_probas_N_K_C, n_instances, max_num_samples, device=device
    )
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates

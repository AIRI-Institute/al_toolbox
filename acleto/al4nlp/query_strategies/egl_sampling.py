import logging
from typing import Union

import numpy as np
from datasets.arrow_dataset import Dataset

from .al_strategy_utils import take_idx
from ..utils.get_gradient_lengths import compute_egl_scores
from ..utils.transformers_dataset import TransformersDataset

log = logging.getLogger()


def egl_sampling(
    model, X_pool: Union[Dataset, TransformersDataset], n_instances: int, **egl_kwargs,
):
    kwargs = dict(
        # Necessary
        model=model,
        dataloader_or_data=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        tokenizer=model.tokenizer,
        task=model.task,
        text_name=model.data_config["text_name"],
        label_name=model.data_config["label_name"],
    )

    scores = compute_egl_scores(**kwargs)
    query_idx = np.argpartition(-scores, n_instances)[:n_instances]

    query = take_idx(X_pool, query_idx)

    uncertainty_estimates = scores

    return query_idx, query, uncertainty_estimates

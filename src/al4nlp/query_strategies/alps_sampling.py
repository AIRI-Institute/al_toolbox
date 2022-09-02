from typing import Union

import numpy as np
import torch
from datasets.arrow_dataset import Dataset

from .al_strategy_utils import (
    take_idx,
    calculate_alps_scores,
)
from ..utils.cluster_utils import kmeans
from ..utils.transformers_dataset import TransformersDataset


def alps_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **alps_kwargs,
):
    """
    Employs the surprisal embedding of w, which is obtained from the likelihoods of masked tokens from
    pre-trained language models. Doesn't require fine-tuning. https://aclanthology.org/2020.emnlp-main.637/
    """
    logits = model.predict_logits(X_pool)

    kwargs = dict(
        # Necessary
        model_wr=model,
        dataloader_or_data=X_pool,
        # General
        data_is_tokenized=False,
        # data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size // 8,
        to_numpy=True,
        logits=logits,
        # train_probas=train_logits,
        # data_is_tokenized=data_is_tokenized,
        tokenizer=model.tokenizer,
        task=model.task,
        text_name=model.data_config["text_name"],
        label_name=model.data_config["label_name"],
    )
    scores_or_vectors = calculate_alps_scores(**kwargs)
    vectors = torch.nn.functional.normalize(scores_or_vectors)
    clustering = kmeans

    query_idx = np.array(clustering(vectors, k=n_instances))

    query = take_idx(X_pool, query_idx)

    # Uncertainty estimates are not defined for ALPS
    uncertainty_estimates = np.zeros(len(X_pool))

    return query_idx, query, uncertainty_estimates

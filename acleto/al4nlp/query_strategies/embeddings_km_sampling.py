import logging
from typing import Union

import numpy as np
from datasets.arrow_dataset import Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from .al_strategy_utils import take_idx
from ..utils.get_embeddings import get_embeddings
from ..utils.transformers_dataset import TransformersDataset

log = logging.getLogger()


def embeddings_km_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    data_is_tokenized=False,
    batch_size=100,
    data_config=None,
    **embeddings_km_kwargs,
):
    """
    Employs the surprisal embedding of w, which is obtained from the likelihoods of masked tokens from
    pre-trained language models. Doesn't require fine-tuning. https://aclanthology.org/2020.emnlp-main.637/
    """
    data_config = data_config if data_config is not None else model.data_config
    kwargs = dict(
        # General
        model=model.model,
        batch_size=batch_size,
        to_numpy=True,
        data_is_tokenized=data_is_tokenized,
        tokenizer=model.tokenizer,
        task=model.task,
        text_name=data_config["text_name"],
        label_name=data_config["label_name"],
    )
    embeddings = get_embeddings(dataloader_or_data=X_pool, **kwargs)
    # embeddings = torch.nn.functional.normalize(embeddings)
    embeddings = normalize(embeddings)

    km = KMeans(n_clusters=n_instances)
    km.fit(embeddings)

    query_idx = get_nearest_to_centers(km.cluster_centers_, embeddings)

    query = take_idx(X_pool, query_idx)

    # Uncertainty estimates are not defined for BERT-KM
    uncertainty_estimates = np.zeros(len(X_pool))

    return query_idx, query, uncertainty_estimates


def get_nearest_to_centers(centers, vectors):
    sim = np.matmul(centers, vectors.T)
    return sim.argmax(axis=1)

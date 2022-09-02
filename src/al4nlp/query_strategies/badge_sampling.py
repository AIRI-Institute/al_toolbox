import logging
import pdb
from typing import Union

import numpy as np
from datasets.arrow_dataset import Dataset
from scipy import stats
from sklearn.metrics import pairwise_distances

from .al_strategy_utils import (
    take_idx,
    calculate_badge_scores,
)
from ..utils.transformers_dataset import TransformersDataset

log = logging.getLogger()


def badge_sampling(
    model,
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    X_train: Union[Dataset, TransformersDataset],
    **badge_kwargs,
):
    """
    Measures uncertainty as the gradient magnitude with respect to parameters in the final (output) layer,
    which is computed using the most likely label according to the model. To capture diversity, collect a
    batch of examples where these gradients span a diverse set of directions. https://arxiv.org/abs/1906.03671
    """
    logits = model.predict_logits(X_pool)

    kwargs = dict(
        # Necessary
        model_wrapper=model,
        data_test=X_pool,
        # General
        data_is_tokenized=False,
        data_config=None,
        batch_size=model._batch_size_kwargs.eval_batch_size,
        to_numpy=False,
        logits=logits,
    )

    vectors = calculate_badge_scores(**kwargs).cpu().detach().numpy()

    query_idx = np.array(init_centers(vectors, k=n_instances))

    query = take_idx(X_pool, query_idx)

    # Uncertainty estimates are not defined for BADGE
    uncertainty_estimates = np.zeros(len(X_pool))

    return query_idx, query, uncertainty_estimates


def init_centers(x, k):
    ind = np.argmax([np.linalg.norm(s, 2) for s in x])
    mu = [x[ind]]
    inds_all = [ind]
    cent_inds = [0.0] * len(x)
    cent = 0
    while len(mu) < k:
        if len(mu) == 1:
            d2 = pairwise_distances(x, mu).ravel().astype(float)
        else:
            new_d = pairwise_distances(x, [mu[-1]]).ravel().astype(float)
            for i in range(len(x)):
                if d2[i] > new_d[i]:
                    cent_inds[i] = cent
                    d2[i] = new_d[i]
        if sum(d2) == 0.0:
            pdb.set_trace()
        d2 = d2.ravel().astype(float)
        d_dist = (d2 ** 2) / sum(d2 ** 2)
        custom_dist = stats.rv_discrete(
            name="custm", values=(np.arange(len(d2)), d_dist)
        )
        ind = custom_dist.rvs(size=1)[0]
        while ind in inds_all:
            ind = custom_dist.rvs(size=1)[0]
        mu.append(x[ind])
        inds_all.append(ind)
        cent += 1
    return inds_all

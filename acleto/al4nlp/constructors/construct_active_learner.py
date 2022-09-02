import sys
from copy import deepcopy
from functools import partial
from importlib import import_module
from pathlib import Path

from small_text.query_strategies import (
    LeastConfidence,
    PredictionEntropy,
)

from ..active_learner import ActiveLearner
from ..pool_subsampling_strategies import (
    ups_subsampling,
    random_subsampling,
    naive_subsampling,
)
from ..query_strategies.actune_sampling import actune_sampling
from ..query_strategies.al_strategy import (
    mahalanobis_triplet_sampling,
    triplet_sampling,
    mahalanobis_filtering_sampling,
    ddu_sampling,
    logits_lc_sampling,
    margin_sampling,
    oracle_sampling,
    hybrid_sampling,
    ssal_sampling,
    ddu_sampling_cv,
)
from ..query_strategies.alps_sampling import alps_sampling
from ..query_strategies.badge_sampling import badge_sampling
from ..query_strategies.bait_sampling import bait_sampling
from ..query_strategies.bald_sampling import bald_sampling
from ..query_strategies.batchbald_sampling import batchbald_sampling
from ..query_strategies.breaking_ties_sampling import breaking_ties_sampling
from ..query_strategies.cal_sampling import cal_sampling
from ..query_strategies.cluster_margin_sampling import cluster_margin_sampling
from ..query_strategies.coreset_sampling import coreset_sampling
from ..query_strategies.egl_sampling import egl_sampling
from ..query_strategies.embeddings_km_sampling import embeddings_km_sampling
from ..query_strategies.entropy_sampling import entropy_sampling
from ..query_strategies.lc_sampling import lc_sampling
from ..query_strategies.mahalanobis_sampling import mahalanobis_sampling
from ..query_strategies.mnlp_sampling import mnlp_sampling
from ..query_strategies.random_sampling import random_sampling
from ..query_strategies.strategy_wrappers.small_text_sampling import small_text_sampling

QUERY_STRATEGIES = {
    # Classification strategies
    "random": partial(random_sampling, select_by_number_of_tokens=False),
    "entropy": entropy_sampling,
    "lc": lc_sampling,
    "ssal": ssal_sampling,
    "logits_lc": logits_lc_sampling,
    "br_ties": breaking_ties_sampling,
    "mahalanobis": mahalanobis_sampling,
    "mahalanobis_triplet": mahalanobis_triplet_sampling,
    "mahalanobis_filtering": mahalanobis_filtering_sampling,
    "triplet": triplet_sampling,
    "ddu": ddu_sampling,
    "margin": margin_sampling,
    "cal": cal_sampling,
    "oracle": oracle_sampling,
    "cluster_margin": cluster_margin_sampling,
    "hybrid": hybrid_sampling,
    "alps": alps_sampling,
    "coreset": coreset_sampling,
    "bait": bait_sampling,
    "badge": badge_sampling,
    "emb_km": embeddings_km_sampling,
    "egl": egl_sampling,
    "actune_lc": partial(actune_sampling, query_strategy=lc_sampling),
    "actune_ent": partial(actune_sampling, query_strategy=entropy_sampling),
    "actune_cal": partial(actune_sampling, query_strategy=cal_sampling),
    "actune_mahalanobis": partial(actune_sampling, query_strategy=mahalanobis_sampling),

    "small-text_lc": partial(small_text_sampling, small_text_strategy=LeastConfidence),
    "small-text_ent": partial(small_text_sampling, small_text_strategy=PredictionEntropy),
    # NER strategies
    "mnlp_tokens": partial(mnlp_sampling, select_by_number_of_tokens=True),
    "mnlp_samples": partial(mnlp_sampling, select_by_number_of_tokens=False),
    "random_tokens": partial(random_sampling, select_by_number_of_tokens=True),
    "random_samples": partial(random_sampling, select_by_number_of_tokens=False),
    # BALD
    "bald": partial(
        bald_sampling, select_by_number_of_tokens=False, only_head_dropout=False
    ),
    "bald_head": partial(
        bald_sampling, select_by_number_of_tokens=False, only_head_dropout=True
    ),
    "bald_tokens": partial(
        bald_sampling, select_by_number_of_tokens=True, only_head_dropout=False
    ),
    "bald_samples": partial(
        bald_sampling, select_by_number_of_tokens=False, only_head_dropout=False
    ),
    "bald_tokens_head": partial(
        bald_sampling, select_by_number_of_tokens=True, only_head_dropout=True
    ),
    "bald_samples_head": partial(
        bald_sampling, select_by_number_of_tokens=False, only_head_dropout=True
    ),
    # BatchBald
    "batchbald": partial(
        batchbald_sampling, select_by_number_of_tokens=False, only_head_dropout=False
    ),
    "batchbald_head": partial(
        batchbald_sampling, select_by_number_of_tokens=False, only_head_dropout=True
    ),
    "batchbald_tokens": partial(
        batchbald_sampling, select_by_number_of_tokens=True, only_head_dropout=False
    ),
    "batchbald_samples": partial(
        batchbald_sampling, select_by_number_of_tokens=False, only_head_dropout=False
    ),
    "batchbald_tokens_head": partial(
        batchbald_sampling, select_by_number_of_tokens=True, only_head_dropout=True
    ),
    "batchbald_samples_head": partial(
        batchbald_sampling, select_by_number_of_tokens=False, only_head_dropout=True
    ),
    # CV strategies
    "ddu_cv": ddu_sampling_cv,
}

SAMPLING_STRATEGIES = {
    "ups": ups_subsampling,
    "random": random_subsampling,
    "naive": naive_subsampling,
}


def construct_active_learner(model, config, initial_data, log_dir: str or Path, original_pool_size=None):

    # TODO: rewrite using `split_by_tokens` as `strategy_kwargs`
    initial_data_copy = deepcopy(initial_data)
    use_subsampling = config.sampling_type is not None
    postfix = ""
    if ("split_by_tokens" in config) and (config.split_by_tokens):
        postfix += "_tokens"
    elif "split_by_tokens" in config:  # avoid adding "_samples" for classification
        postfix += "_samples"

    if config.strategy == "actune":
        postfix += getattr(config, "base_strategy", "_lc")

    if config.strategy == "small-text":
        postfix += getattr(config, "small_text_strategy", "_lc")

    query_strategy = QUERY_STRATEGIES.get(f"{config.strategy}{postfix}")
    # In this case, we assume that `config.strategy` refers to the path of the file with the strategy
    if query_strategy is None:
        query_strategy = _get_strategy_from_path(config.strategy)
    sampling_strategy = SAMPLING_STRATEGIES.get(config.sampling_type)
    if use_subsampling and sampling_strategy is None:
        sampling_strategy = _get_strategy_from_path(config.sampling_type)
    sampling_kwargs = {
        "gamma_or_k_confident_to_save": config.gamma_or_k_confident_to_save,
        "T": config.T,
    }
    strategy_kwargs = config.strategy_kwargs

    learner = ActiveLearner(
        estimator=model,
        query_strategy=query_strategy,
        train_data=initial_data_copy,
        strategy_kwargs=strategy_kwargs,
        sampling_strategy=sampling_strategy,
        sampling_kwargs=sampling_kwargs,
        log_dir=log_dir,
        original_pool_size=original_pool_size
    )

    return learner


def _get_strategy_from_path(strategy_path):
    if (not strategy_path.endswith("/")) and ("/" in strategy_path):
        path = strategy_path[: strategy_path.rindex("/")]
    elif "/" in strategy_path:
        path = strategy_path[: strategy_path[:-2].rindex("/")]
    else:
        path = "./"
    strategy_name = strategy_path.split("/")[-1] or strategy_path.split("/")[-2]
    if strategy_name.endswith(".py"):
        strategy_name = strategy_name[:-3]
    sys.path.append(path)

    strategy_func = getattr(import_module(strategy_name), strategy_name, None)
    if strategy_func is None:
        strategy_func = getattr(
            import_module(strategy_name), strategy_name + "_sampling"
        )
    return strategy_func

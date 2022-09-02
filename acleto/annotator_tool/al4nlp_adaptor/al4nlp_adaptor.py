import logging
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from ...al4nlp.constructors.construct_active_learner import construct_active_learner
from ...al4nlp.query_strategies.al_strategy_utils import (
    take_idx,
)
from ...al4nlp.utils.general import json_dump
from ...al4nlp.utils.transformers_dataset import TransformersDataset

logger = logging.getLogger()


class AdaptorAl4Nlp:
    def __init__(
        self,
        config,
        model,
        label2id,
        task="ner",
        log_dir="./",
        strategy_kwargs=None
    ):      
        self.model = model
        self.task = task
        self.config = config
        self.label2id = label2id
        self.learner = None
        self._create_time_dict(log_dir)
        self.strategy_kwargs = strategy_kwargs if not None else {}
        
        self.X_pool = None

    def start(self, X_labeled, y_labeled, X_unlabeled, y_unlabeled=None):
        labeled = TransformersDataset(
            {self.config.data.text_name: X_labeled,
             self.config.data.label_name: y_labeled}
        )
        self.X_pool = TransformersDataset({self.config.data.text_name: X_unlabeled})
        self.strings = pd.Series([" ".join(inst['tokens']) for inst in self.X_pool], index=list(range(len(self.X_pool))))
        self.y_unlabeled = y_unlabeled
        
        logger.info(f"Running first scoring iteration")
        self.learner = construct_active_learner(self.model, self.config.al, labeled, "/") # TODO: config
        
    def remove_duplicates(self, unlabeled_pool_indexes):
        unlabeled_strings = self.strings[unlabeled_pool_indexes]
        unlabeled_strings = unlabeled_strings.drop_duplicates()
        return unlabeled_strings.index
        
    def choose_samples_for_annotation(self, n_instances):
        unlabeled_pool_indexes = self.get_unlabeled_pool_indexes()
        unlabeled_pool_indexes = self.remove_duplicates(unlabeled_pool_indexes)
        unlabeled_pool = self.X_pool.select(unlabeled_pool_indexes)

        logger.info(f'Requesting {n_instances} instances from the pool of size {len(unlabeled_pool)}')
        
        start_time = time()
        (
            query_idx,
            query_instance,
            uncertainty_estimates,
            *query_meta,
        ) = self.learner.query(
            unlabeled_pool,
            n_instances=n_instances,
            indexes=unlabeled_pool_indexes,
            **self.strategy_kwargs
        )
        query_time = time() - start_time
        self._add_obs_to_time_dict("query", query_time)
        
        logger.info(f'Queried: {query_idx.size}')

        real_query_idx = unlabeled_pool_indexes[query_idx]
        return real_query_idx

    def convert_markup_to_instances(self, indexes, y):
        query_instance = take_idx(self.X_pool, indexes) if len(indexes) > 0 else []

        if self.task == "ner" and query_instance:
            ner_tags = []
            for example in y:
                ner_tags.append([self.label2id[tag] for tag in example])

            query_instance.add_column("ner_tags", ner_tags)

        if self.task == "cls" and query_instance:
            query_instance.add_column("label", np.array(y, dtype=int))

        return query_instance

    def make_iteration(self, indexes, y):
        query_instance = self.convert_markup_to_instances(np.asarray(indexes), np.asarray(y))

        if query_instance:
            self.learner._add_data(query_instance)
            self.learner._fit_estimator()

    def _create_time_dict(self, log_dir: str or Path):
        self._time_dict = {"fit": [], "query": []}
        self._time_dict_path = Path(log_dir) / "al_learner_time_dict.json"
        if not Path(log_dir).exists():
            Path(log_dir).mkdir()
        json_dump(self._time_dict, self._time_dict_path)

    def _add_obs_to_time_dict(self, step: str, value: float):
        self._time_dict[step].append(value)
        json_dump(self._time_dict, self._time_dict_path)

    def get_unlabeled_pool_indexes(self):
        y_arr = pd.Series(self.y_unlabeled)
        return np.asarray(y_arr[y_arr.isnull()].index)

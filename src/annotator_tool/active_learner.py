import logging

import numpy as np
import pandas as pd

logger = logging.getLogger()


class ActiveLearner:
    """A class that implements the active learning logic."""

    def __init__(
        self,
        active_learn_alg, 
        
        X_labeled_dataset,
        y_labeled_dataset,
        X_unlabeled_dataset,
        y_unlabeled_dataset=None,
        
        model_evaluate=None,
        X_test_dataset=None,
        y_test_dataset=None,
        eval_metrics=None,
        
        rnd_start_steps=0,
        n_instances=10
    ):
        """
        ActiveLearner constructor.

        Args:
            active_learn_alg (functor): query strategy wrapper.
            X_full_dataset (np.array or sparse matrix): feature matrix.
            y_dtype: type of y labels.
            y_full_dataset (np.array): known answers (e.g., None -- unknown, True -- positive class, False -- negative class)
            model_evaluate: the model that will be evaluated on the holdout.
            X_test_dataset: feature matrix for testing via holdout.
            y_test_dataset: y labels for testing via holdout.
            eval_metrics (list): list of sklearn evaluation metrics.
            rnd_start_steps: AL will can make several seed steps by choosing random samples (without model suggestions).
            logger (logging.Logger): the object for logging.

        """
        super().__init__()
        
        self._X_unlabeled_dataset = X_unlabeled_dataset
        self._X_labeled_dataset = X_labeled_dataset
        self._y_labeled_dataset = y_labeled_dataset
        if y_unlabeled_dataset is not None:
            self._y_unlabeled_dataset  = y_unlabeled_dataset 
        else:
            self._y_unlabeled_dataset  = [None] * len(
                self._X_unlabeled_dataset
            )

        self._active_learn_alg = active_learn_alg

        self._X_test_dataset = X_test_dataset
        self._y_test_dataset = y_test_dataset
        self._model_evaluate = model_evaluate
        self._eval_metrics = eval_metrics

        self._iteration_num = 0
        self._rnd_start_steps = rnd_start_steps
        
        self.n_instances = n_instances

    def _select_unannotated(self, labels):
        return np.where([(e is None) for e in labels])[0]

    def start(self):
        self._active_learn_alg.start(
            self._X_labeled_dataset, self._y_labeled_dataset, 
            self._X_unlabeled_dataset, self._y_unlabeled_dataset)

    def choose_random_sample_for_annotation(self, number):
#         if len(np.where([(e is None) for e in self._y_unlabeled_dataset])[0]) == len(
#             self._X_unlabeled_dataset
#         ):
            
#         else:
#             ids_array = np.arange(len(self._X_unlabeled_dataset))
# TODO: fix
        ids_array = self._select_unannotated(self._y_unlabeled_dataset)
        return np.random.choice(ids_array, size=number, replace=False)

    def choose_samples_for_annotation(self):
        if self._iteration_num < self._rnd_start_steps:
            return self.choose_random_sample_for_annotation(self.n_instances)
        else:
            return self._active_learn_alg.choose_samples_for_annotation(self.n_instances)

    def evaluate(self, fit_model=True):
        if self._model_evaluate is None:
            return None

        if fit_model:
            selector = [n for n, e in enumerate(self._y_full_dataset) if e is not None]
            y_fit = [self._y_full_dataset[i] for i in selector]
            X_fit = [self._X_full_dataset[i] for i in selector]
            logger.info("Number of training samples: {}".format(len(y_fit)))

            self._model_evaluate.fit(X_fit, y_fit)

        preds = self._model_evaluate.predict(self._X_test_dataset)
        return {
            metric.__name__: metric(preds, self._y_test_dataset)
            for metric in self._eval_metrics
        }

    def get_annotation(self):
        return self._y_unlabeled_dataset

    def make_iteration(self, indexes, answers):
        self.add_answers(indexes, answers)
        res = self._active_learn_alg.make_iteration(indexes, answers)
        self._iteration_num += 1
        return res

    def add_answers(self, indexes, answers):
        answers = pd.Series(answers, index=indexes, dtype='object')
        answers = answers[answers.notnull()]
        logger.info(f"Iteration Answers: {answers.to_dict()}")
        for num, i in enumerate(answers.index):
            self._y_unlabeled_dataset[i] = answers[i]
            
    def is_busy(self):
        return False

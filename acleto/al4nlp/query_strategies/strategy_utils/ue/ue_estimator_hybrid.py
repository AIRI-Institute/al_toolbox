import copy
import logging
import time

import numpy as np
from scipy.stats import rankdata

from .mahalanobis_distance import (
    mahalanobis_distance,
    mahalanobis_distance_relative,
    mahalanobis_distance_marginal,
    compute_centroids,
    compute_covariance,
)
from .utils_heads import (
    ElectraClassificationHeadIdentityPooler,
    BertClassificationHeadIdentityPooler,
    ElectraNERHeadIdentityPooler,
)
from .utils_inference import is_custom_head, unpad_features, pad_scores

log = logging.getLogger()


def entropy(x):
    return np.sum(-x * np.log(np.clip(x, 1e-8, 1)), axis=-1)


class UeEstimatorHybrid:
    def __init__(self, cls, ue_args, config, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset

    def __call__(self, X):
        return self._predict_with_fitted_cov(X)

    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model

        log.info(
            "****************Start fitting covariance and centroids **************"
        )

        if y is None:
            y = self._extract_labels(X)

        self._replace_model_head()
        X_features = self._extract_features(X)

        self.class_cond_centroids = self._fit_centroids(X_features, y)
        self.class_cond_covarince = self._fit_covariance(X_features, y)

        self.fit_all_md_versions = (
            "fit_all_md_versions" in self.ue_args.keys()
            and self.ue_args["fit_all_md_versions"]
        )
        if self.fit_all_md_versions:
            self.train_centroid = self._fit_centroids(X_features, y, class_cond=False)
            self.train_covarince = self._fit_covariance(X_features, y, class_cond=False)

        log.info("**************Done.**********************")

    def _fit_covariance(self, X, y, class_cond=True):
        if class_cond:
            return compute_covariance(self.class_cond_centroids, X, y, class_cond)
        return compute_covariance(self.train_centroid, X, y, class_cond)

    def _fit_centroids(self, X, y, class_cond=True):
        return compute_centroids(X, y, class_cond)

    def _replace_model_head(self):
        log.info("Change classifier to Identity Pooler")
        cls = self.cls
        model = self.cls._auto_model
        self.old_head = copy.deepcopy(model.classifier)

        if is_custom_head(model):
            model.classifier = ElectraClassificationHeadIdentityPooler(model.classifier)
        else:
            model.classifier = BertClassificationHeadIdentityPooler(model.classifier)

    def _return_head(self):
        self.cls._auto_model.classifier = self.old_head
        log.info("Change Identity Pooler to classifier")

    def _extract_labels(self, X):
        return np.asarray([example["labels"] for example in X])

    def _extract_features(self, X):
        cls = self.cls
        model = self.cls._auto_model
        X = X.remove_columns("labels")

        X_features = cls.predict(X, apply_softmax=False, return_preds=True)[0]
        return X_features

    def _predict_with_fitted_cov(self, X):
        cls = self.cls
        model = self.cls._auto_model

        log.info(
            "****************Compute MD with fitted covariance and centroids **************"
        )

        self._return_head()
        probs = cls.predict(X, apply_softmax=True, return_preds=False)[0]
        unc_entropy = entropy(probs)
        self._replace_model_head()
        self.md_threshold = (
            self.ue_args["md_threshold"]
            if "md_threshold" in self.ue_args.keys()
            else 0.2
        )

        start = time.time()
        X_features = self._extract_features(X)

        end = time.time()

        eval_results = {}

        md, inf_time = mahalanobis_distance(
            None, None, X_features, self.class_cond_centroids, self.class_cond_covarince
        )

        sum_inf_time = inf_time + (end - start)

        n_preds = len(X)
        n_lowest = int(n_preds * self.md_threshold)

        md_rank = rankdata(md)
        unc_entropy_rank = rankdata(unc_entropy[md_rank < n_lowest])
        md_rank[md_rank < n_lowest] = unc_entropy_rank

        eval_results["mahalanobis_distance"] = md_rank.tolist()
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")

        if self.fit_all_md_versions:
            md_relative = mahalanobis_distance_relative(
                None, None, X_features, self.train_centroid, self.train_covarince
            )

            md_marginal = mahalanobis_distance_marginal(
                None,
                None,
                X_features,
                self.class_cond_centroids,
                self.class_cond_covarince,
                self.train_centroid,
                self.train_covarince,
            )

            eval_results["mahalanobis_distance_relative"] = md_relative.tolist()
            eval_results["mahalanobis_distance_marginal"] = md_marginal.tolist()

        log.info("**************Done.**********************")
        return eval_results


class UeEstimatorMahalanobisNer:
    def __init__(self, cls, ue_args, config, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset

    def __call__(self, X, y):
        return self._predict_with_fitted_cov(X, y)

    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model

        log.info(
            "****************Start fitting covariance and centroids **************"
        )

        if y is None:
            y, y_shape = self._extract_labels(X)

        self._replace_model_head()
        X_features = self._extract_features(X)
        X_features, y = unpad_features(X_features, y)

        self.class_cond_centroids = self._fit_centroids(X_features, y)
        self.class_cond_covarince = self._fit_covariance(X_features, y)

        self.fit_all_md_versions = (
            "fit_all_md_versions" in self.ue_args.keys()
            and self.ue_args["fit_all_md_versions"]
        )

        if self.fit_all_md_versions:
            self.train_centroid = self._fit_centroids(X_features, y, class_cond=False)
            self.train_covarince = self._fit_covariance(X_features, y, class_cond=False)

        log.info("**************Done.**********************")

    def _fit_covariance(self, X, y, class_cond=True):
        if class_cond:
            return compute_covariance(self.class_cond_centroids, X, y, class_cond)
        return compute_covariance(self.train_centroid, X, y, class_cond)

    def _fit_centroids(self, X, y, class_cond=True):
        return compute_centroids(X, y, class_cond)

    def _replace_model_head(self):
        log.info("Change classifier to Identity Pooler")
        cls = self.cls
        model = self.cls._auto_model

        if is_custom_head(model):
            model.classifier = ElectraNERHeadIdentityPooler(model.classifier)
        else:
            model.classifier = BertClassificationHeadIdentityPooler(model.classifier)

    def _extract_labels(self, X):
        y = np.asarray([example["labels"] for example in X])
        y_shape = y.shape

        return y.reshape(-1), y_shape

    def _extract_features(self, X):
        cls = self.cls
        model = self.cls._auto_model

        try:
            X = X.remove_columns("labels")
        except:
            X.dataset = X.dataset.remove_columns("labels")

        X_features = cls.predict(X, apply_softmax=False, return_preds=False)[0]
        X_features = X_features.reshape(-1, X_features.shape[-1])

        return X_features

    def _predict_with_fitted_cov(self, X, y):
        cls = self.cls
        model = self.cls._auto_model

        log.info(
            "****************Compute MD with fitted covariance and centroids **************"
        )

        start = time.time()

        y_pad, y_shape = self._extract_labels(X)
        X_features = self._extract_features(X)
        X_features, y = unpad_features(X_features, y_pad)

        end = time.time()

        eval_results = {}

        md, inf_time = mahalanobis_distance(
            None, None, X_features, self.class_cond_centroids, self.class_cond_covarince
        )

        md = pad_scores(md, np.asarray(y_pad).reshape(y_shape), y_pad)

        sum_inf_time = inf_time + (end - start)
        eval_results["mahalanobis_distance"] = md.tolist()
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")

        if self.fit_all_md_versions:
            md_relative = mahalanobis_distance_relative(
                None, None, X_features, self.train_centroid, self.train_covarince
            )
            md_relative = pad_scores(
                md_relative, np.asarray(y_pad).reshape(y_shape), y_pad
            )

            md_marginal = mahalanobis_distance_marginal(
                None,
                None,
                X_features,
                self.class_cond_centroids,
                self.class_cond_covarince,
                self.train_centroid,
                self.train_covarince,
            )
            md_relative = pad_scores(
                md_relative, np.asarray(y_pad).reshape(y_shape), y_pad
            )

            eval_results["mahalanobis_distance_relative"] = md_relative.tolist()
            eval_results["mahalanobis_distance_marginal"] = md_marginal.tolist()

        log.info("**************Done.**********************")
        return eval_results

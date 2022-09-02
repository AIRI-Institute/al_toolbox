import numpy as np


def naive_subsampling(uncertainty_estimates, gamma_or_k_confident_to_save, **kwargs):
    if isinstance(gamma_or_k_confident_to_save, float):
        gamma_or_k_confident_to_save = int(
            gamma_or_k_confident_to_save * len(uncertainty_estimates)
        )
    argsort = np.argsort(-uncertainty_estimates)
    return argsort[:gamma_or_k_confident_to_save]

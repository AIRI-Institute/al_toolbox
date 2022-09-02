import numpy as np


def random_subsampling(uncertainty_estimates, gamma_or_k_confident_to_save, **kwargs):
    length = len(uncertainty_estimates)
    if isinstance(gamma_or_k_confident_to_save, float):
        gamma_or_k_confident_to_save = int(gamma_or_k_confident_to_save * length)
    if gamma_or_k_confident_to_save >= length:
        return np.arange(length)
    return np.random.choice(
        np.arange(length), gamma_or_k_confident_to_save, replace=False
    )

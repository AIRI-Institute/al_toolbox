import numpy as np


def get_ups_sampling_probas(argsort, gamma, T):
    ranks = argsort.argsort() / len(argsort)
    return np.exp(-np.maximum(0, ranks - gamma) / np.maximum(T, 1e-8))


def sample_idxs(sampling_probas):
    to_select = []
    for i in range(len(sampling_probas)):
        proba_to_choose_i = sampling_probas[i]
        if np.random.uniform() < proba_to_choose_i:
            to_select.append(i)
    return np.array(to_select)


def ups_subsampling(uncertainty_estimates, gamma_or_k_confident_to_save, T):
    if isinstance(gamma_or_k_confident_to_save, int):
        gamma_or_k_confident_to_save /= len(uncertainty_estimates)
    argsort = np.argsort(-uncertainty_estimates)
    sampling_probas = get_ups_sampling_probas(argsort, gamma_or_k_confident_to_save, T)
    return sample_idxs(sampling_probas)

"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    n, d = X.shape  # Number of data points and dimensions
    k = mixture.mu.shape[0]  # Number of clusters

    # Initialize the responsibilities (posterior probabilities) matrix
    log_resp = np.zeros((n, k))

    for j in range(k):
        # Calculate the log probability density of each point in cluster j
        diff = X - mixture.mu[j]  # Difference from the mean
        squared_mahalanobis_distance = np.sum((diff ** 2) / mixture.var[j], axis=1)
        log_prob = -0.5 * (d * np.log(2 * np.pi) + d * np.log(mixture.var[j]) + squared_mahalanobis_distance)
        log_resp[:, j] = log_prob + np.log(mixture.p[j])

    # Manually calculate log-sum-exp to avoid numerical underflow
    max_log_resp = np.max(log_resp, axis=1, keepdims=True)
    sum_exp_log_resp = np.sum(np.exp(log_resp - max_log_resp), axis=1, keepdims=True)
    logsumexp_resp = np.log(sum_exp_log_resp) + max_log_resp

    # Normalize to get the actual responsibilities
    log_resp -= logsumexp_resp

    # Calculate the log-likelihood
    log_likelihood = np.sum(logsumexp_resp)

    return np.exp(log_resp), log_likelihood


    # return np.exp(log_resp), log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape  # Number of data points and dimensions
    _, k = post.shape  # Number of clusters

    # Calculate the new weights
    new_weights = np.sum(post, axis=0) / n

    # Calculate the new means
    new_means = np.dot(post.T, X) / np.sum(post, axis=0)[:, np.newaxis]

    # Calculate the new variances
    new_vars = np.zeros(k)
    for j in range(k):
        diff = X - new_means[j]
        weighted_diff = diff ** 2 * post[:, j, np.newaxis]
        new_vars[j] = np.sum(weighted_diff) / (np.sum(post[:, j]) * d)

    return GaussianMixture(new_means, new_vars, new_weights)



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_log_likelihood = None
    log_likelihood = -np.inf  # Initialize log-likelihood

    while True:
        # E-step
        post, log_likelihood = estep(X, mixture)

        # Check for convergence
        if prev_log_likelihood is not None and log_likelihood - prev_log_likelihood <= 1e-6 * abs(log_likelihood):
            break

        prev_log_likelihood = log_likelihood

        # M-step
        mixture = mstep(X, post)

    return mixture, post, log_likelihood



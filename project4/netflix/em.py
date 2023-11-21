"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
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

    # Compute log-sum-exp to avoid numerical underflow
    logsumexp_resp = logsumexp(log_resp, axis=1).reshape(-1, 1)
    
    # Normalize to get the actual responsibilities
    log_resp -= logsumexp_resp
    
    # Calculate the log-likelihood
    log_likelihood = np.sum(logsumexp_resp)

    return np.exp(log_resp), log_likelihood



    


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

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
        if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < 1e-6:
            break

        prev_log_likelihood = log_likelihood

        # M-step
        mixture = mstep(X, post)

    return mixture, post, log_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError

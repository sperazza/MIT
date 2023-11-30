"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from copy import deepcopy


def manual_logsumexp(log_probs, axis=None):
    # Find the maximum log-probability to factor out
    max_log_prob = np.max(log_probs, axis=axis, keepdims=True)
    # Stabilize the input to exp by subtracting the maximum log-probability
    stabilized_log_probs = log_probs - max_log_prob
    # Sum the exponentials of the stabilized log-probabilities
    sum_exp = np.sum(np.exp(stabilized_log_probs), axis=axis, keepdims=True)
    # Take the log and add back the factor we factored out
    return max_log_prob + np.log(sum_exp)

# Replace the following line in your estep function:
# logsumexp_resp = logsumexp(log_resp, axis=1).reshape(-1, 1)
# With:
#logsumexp_resp = manual_logsumexp(log_resp, axis=1)


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
        squared_mahalanobis_distance = np.sum((diff ** 2.0) / mixture.var[j], axis=1)
        log_prob = -0.5 * (d * np.log(2.0 * np.pi) + d * np.log(mixture.var[j]) + squared_mahalanobis_distance)
        log_resp[:, j] = log_prob + np.log(mixture.p[j])

    # Compute log-sum-exp to avoid numerical underflow
    logsumexp_resp = logsumexp(log_resp, axis=1).reshape(-1, 1)
    #logsumexp_resp = manual_logsumexp(log_resp, axis=1)
    
    # Normalize to get the actual responsibilities
    log_resp -= logsumexp_resp
    
    # Calculate the log-likelihood
    log_likelihood = np.sum(logsumexp_resp)

    return np.exp(log_resp), log_likelihood



    

#mstep that was accepted by autograder
# def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
#           min_variance: float = .25) -> GaussianMixture:
#     """M-step: Updates the gaussian mixture by maximizing the log-likelihood
#     of the weighted dataset

#     Args:
#         X: (n, d) array holding the data, with incomplete entries (set to 0)
#         post: (n, K) array holding the soft counts
#             for all components for all examples
#         mixture: the current gaussian mixture
#         min_variance: the minimum variance for each gaussian

#     Returns:
#         GaussianMixture: the new gaussian mixture
#     """
#     n, d = X.shape  # Number of data points and dimensions
#     _, k = post.shape  # Number of clusters

#     # Calculate the new weights
#     new_weights = np.sum(post, axis=0) / n

#     # Calculate the new means
#     new_means = np.dot(post.T, X) / np.sum(post, axis=0)[:, np.newaxis]

#     # Calculate the new variances
#     new_vars = np.zeros(k)
#     for j in range(k):
#         diff = X - new_means[j]
#         weighted_diff = diff ** 2 * post[:, j, np.newaxis]
#         new_vars[j] = np.sum(weighted_diff) / (np.sum(post[:, j]) * d)

#     return GaussianMixture(new_means, new_vars, new_weights)

# below adds usage of min_variance
def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture, min_variance: float = 0.25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
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
        weighted_diff = diff ** 2.0 * post[:, j, np.newaxis]
        # Compute the variance and apply the minimum variance constraint
        variance = np.sum(weighted_diff) / (np.sum(post[:, j]) * d)
        new_vars[j] = max(variance, min_variance)  # Ensuring variance is not below the minimum

    return GaussianMixture(new_means, new_vars, new_weights)


def run_em(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray,verbose=False) -> Tuple[GaussianMixture, np.ndarray, float]:
    prev_log_likelihood = None
    log_likelihood = -np.inf  # Initialize log-likelihood

    while True:
        # E-step
        post, log_likelihood = estep(X, mixture)

        # Check for convergence
        if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < 1e-6:
            break

        if verbose:
            print(log_likelihood)

        prev_log_likelihood = log_likelihood

        # M-step
        mixture = mstep(X, post, mixture, min_variance=0.25)

    return mixture, post, log_likelihood




def run(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the EM algorithm"""
    # Deep copy X to ensure the original data is not modified
    X_copy = deepcopy(X)
    
    # Initialize previous log-likelihood to negative infinity
    prev_log_likelihood = -np.inf
    # Run the initial E-step
    post, log_likelihood = estep(X_copy, mixture)
    
    # Convergence criteria
    while log_likelihood - prev_log_likelihood > 1e-6 * np.abs(log_likelihood):
        # Store the old log-likelihood
        prev_log_likelihood = log_likelihood
        # M-step
        mixture = mstep(X_copy, post, mixture)
        # E-step
        post, log_likelihood = estep(X_copy, mixture)
    
    return mixture, post, log_likelihood




def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    X_pred = np.copy(X)
    K = mixture.mu.shape[0]

    for i in range(n):
        observed_indices = np.where(X[i] != 0)[0]
        missing_indices = np.where(X[i] == 0)[0]

        if len(missing_indices) == 0:
            continue

        # Initialize arrays to store conditional means and probabilities for each component
        conditional_means = np.zeros((K, len(missing_indices)))
        component_probs = np.zeros(K)

        for k in range(K):
            mu_k = mixture.mu[k]
            var_k = mixture.var[k]
            p_k = mixture.p[k]

            # Compute the conditional mean for the missing values
            # Here we assume a simple model where the conditional mean is equal to the mean of the component
            # for the missing values. This can be adjusted to a more complex model if needed.
            conditional_means[k] = mu_k[missing_indices]

            # Compute the probability of this component for the observed data
            # This is a simplification and assumes independence between dimensions.
            prob = np.prod(1 / np.sqrt(2 * np.pi * var_k) * np.exp(-0.5 * ((X[i, observed_indices] - mu_k[observed_indices])**2) / var_k))
            component_probs[k] = prob * p_k

        # Normalize the component probabilities
        component_probs /= np.sum(component_probs)

        # Calculate the expected values for the missing data as a weighted sum of conditional means
        expected_values = np.dot(component_probs, conditional_means)

        # Fill in the missing values in the row with the expected values
        X_pred[i, missing_indices] = expected_values

    return X_pred



# def run(X: np.ndarray, mixture: GaussianMixture,
#         post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
#     """Runs the mixture model

#     Args:
#         X: (n, d) array holding the data
#         post: (n, K) array holding the soft counts
#             for all components for all examples

#     Returns:
#         GaussianMixture: the new gaussian mixture
#         np.ndarray: (n, K) array holding the soft counts
#             for all components for all examples
#         float: log-likelihood of the current assignment
#     """
#     prev_log_likelihood = None
#     log_likelihood = -np.inf  # Initialize log-likelihood

#     while True:
#         # E-step
#         post, log_likelihood = estep(X, mixture)

#         # Check for convergence
#         if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < 1e-6:
#             break

#         prev_log_likelihood = log_likelihood

#         # M-step
#         mixture = mstep(X, post)

#     return mixture, post, log_likelihood
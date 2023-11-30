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


from scipy.stats import multivariate_normal



import scipy

# def estep(X: np.ndarray, mixture: GaussianMixture, min_variance: float=0.25) -> Tuple[np.ndarray, float]:
#     #print("Entering estep")
#     n, d = X.shape
#     K, _ = mixture.mu.shape
#     post = np.zeros((n, K))  # Posterior probabilities
#     log_likelihood = 0.0

#     for i in range(n):
#         #print(f"Processing data point {i+1}/{n}")
#         mask = X[i, :] != 0
#         masked_X = X[i, mask]
#         log_resps = np.zeros(K)

#         for j in range(K):
#             #print(f"Processing component {j+1}/{K} for data point {i+1}/{n}")
#             mean = mixture.mu[j, mask]
#             cov = np.full(mask.sum(), mixture.var[j] + min_variance)
#             cov_matrix = np.diag(cov)
            
#             try:
#                 log_resps[j] = multivariate_normal.logpdf(masked_X, mean=mean, cov=cov_matrix)
#                 log_resps[j] += np.log(mixture.p[j] + 1e-16)
#             except Exception as e:
#                 print(f"An exception occurred for component {j+1}/{K} for data point {i+1}/{n}: {e}")
#                 continue

#         logsumexp = scipy.special.logsumexp(log_resps)
#         post[i, :] = np.exp(log_resps - logsumexp)
#         log_likelihood += logsumexp
#     #print("Exiting estep")

#     return post, log_likelihood

from scipy.stats import multivariate_normal

def estep(X: np.ndarray, mixture: GaussianMixture, min_variance: float=0.25) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))  # Posterior probabilities
    log_likelihood = 0.0

    # Precompute the covariance matrices for each component
    cov_matrices = [mixture.var[j] * np.eye(d) + np.eye(d) * min_variance for j in range(K)]

    for i in range(n):
        mask = X[i, :] != 0
        masked_X = X[i, mask]
        log_resps = np.zeros(K)

        for j in range(K):
            mean = mixture.mu[j, mask]
            # Use the precomputed covariance matrix for the current component
            cov_matrix = cov_matrices[j][np.ix_(mask, mask)]

            # Calculate logpdf in the log domain to prevent underflow
            log_resps[j] = multivariate_normal.logpdf(masked_X, mean=mean, cov=cov_matrix)
            log_resps[j] += np.log(mixture.p[j] + 1e-16)

        logsumexp = scipy.special.logsumexp(log_resps)
        post[i, :] = np.exp(log_resps - logsumexp)
        log_likelihood += logsumexp

    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = 0.25) -> GaussianMixture:
    """M-step: Updates the Gaussian mixture by maximizing the log-likelihood
    of the weighted dataset.

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current Gaussian mixture
        min_variance: the minimum variance for each Gaussian

    Returns:
        GaussianMixture: the new Gaussian mixture
    """
    #print("Entering  mstep")
    n, d = X.shape
    K, _ = mixture.mu.shape
    new_mu = np.zeros((K, d))
    new_p = np.zeros(K)
    new_var = np.zeros(K)
    
    for k in range(K):
        # Responsibilities for component k
        Nk = np.sum(post[:, k])
        new_p[k] = Nk / n
        
        # Update the means
        for feature in range(d):
            # Only use present values for each feature
            mask = X[:, feature] != 0
            new_mu[k, feature] = np.sum(X[mask, feature] * post[mask, k]) / (np.sum(post[mask, k]) + 1e-16)  # Adding epsilon to avoid division by zero
        
        # Update the variances
        masked_X = X - new_mu[k]
        masked_X **= 2
        var = np.sum(post[:, k, np.newaxis] * masked_X, axis=0) / (Nk + 1e-16)  # Adding epsilon to avoid division by zero
        avg_var = np.mean(var[mask])  # Mean variance across all features
        new_var[k] = max(avg_var, min_variance)  # Apply the min_variance threshold

    #print("Exiting mstep")
    return GaussianMixture(new_mu, new_var, new_p)



#runs matrix completion
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
        print(f"Iteration  log_likelihood: {log_likelihood}, change: {log_likelihood - prev_log_likelihood}")
    
        if not np.isfinite(log_likelihood):
            raise ValueError("Log likelihood is not finite.")
    
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

        conditional_means = np.zeros((K, len(missing_indices)))
        component_probs = np.zeros(K)

        for k in range(K):
            mu_k = mixture.mu[k]
            var_k = mixture.var[k]
            p_k = mixture.p[k]

            conditional_means[k] = mu_k[missing_indices]

            prob = np.prod(1 / np.sqrt(2 * np.pi * var_k) * np.exp(-0.5 * ((X[i, observed_indices] - mu_k[observed_indices])**2) / var_k))
            component_probs[k] = prob * p_k

        component_probs /= np.sum(component_probs)
        expected_values = np.dot(component_probs, conditional_means)
        X_pred[i, missing_indices] = expected_values

    return X_pred


if __name__ == "__main__":
    #from sklearn.mixture import GaussianMixture    
    import common
    X=np.loadtxt("netflix_incomplete.txt")
    X_Gold=np.loadtxt("netflix_complete.txt")

    mixture, post = common.init(X, 12, 3)
    final_mixture, final_post, log_likelihood = run(X, mixture, post)

    rmse=common.rmse(X_Gold, fill_matrix(X, final_mixture))
    print(f"RMSE: {rmse}")


    for k in [1, 12]:
        highest_log_likelihood = float('-inf')
        best_gmm = None
        for seed in [0, 1, 2, 3, 4]:
            mixture, post = common.init(X, k, seed)

            # Call the run function with the initialized values
            final_mixture, final_post, log_likelihood = run(X, mixture, post)

            # gmm = GaussianMixture(n_components=k, init_params='kmeans', random_state=seed)                
            # #gmm = GaussianMixture(n_components=k, random_state=seed)
            # gmm.fit(X)
            # log_likelihood = gmm.score(X) * len(X)  # score returns log-likelihood per sample
            print(f"k={k}, seed={seed}, ll={log_likelihood}")
            if log_likelihood > highest_log_likelihood:
                highest_log_likelihood = log_likelihood
                #best_gmm = gmm

        print(f"k={k}, seed={seed} highest_log_likelihood={highest_log_likelihood}")





# Replace the following line in your estep function:
# logsumexp_resp = logsumexp(log_resp, axis=1).reshape(-1, 1)
# With:
#logsumexp_resp = manual_logsumexp(log_resp, axis=1)


# def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
#     """E-step: Softly assigns each datapoint to a gaussian component

#     Args:
#         X: (n, d) array holding the data, with incomplete entries (set to 0)
#         mixture: the current gaussian mixture

#     Returns:
#         np.ndarray: (n, K) array holding the soft counts
#             for all components for all examples
#         float: log-likelihood of the assignment

#     """
#     n, d = X.shape
#     K, _ = mixture.mu.shape
#     post = np.zeros((n, K))  # Posterior probabilities

#     for j in range(K):
#         # Mask for non-missing values (X != 0)
#         non_missing_mask = (X != 0)
        
#         # Adjusting likelihood calculation for non-missing values
#         diff = np.where(non_missing_mask, X - mixture.mu[j], 0)
#         var = mixture.var[j]
#         likelihood = -0.5 * np.sum(((diff ** 2) / var) * non_missing_mask, axis=1)
#         likelihood -= 0.5 * np.sum(np.log(2 * np.pi * var) * non_missing_mask, axis=1)

#         post[:, j] = likelihood + np.log(mixture.p[j] + 1e-16)

#     logsumexp_post = logsumexp(post, axis=1)
#     log_likelihood = np.sum(logsumexp_post)
#     post = np.exp(post - logsumexp_post[:, np.newaxis])

#     return post, log_likelihood



# def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
#     n, d = X.shape
#     K, _ = mixture.mu.shape
#     post = np.zeros((n, K))
#     total_log_likelihood = 0.0

#     for i in range(n):
#         mask = X[i, :] != 0
#         masked_X = X[i, mask]
#         log_likelihoods = np.zeros(K)

#         for j in range(K):
#             # Use the masked array and mixture parameters to calculate the logpdf
#             mean = mixture.mu[j, mask]
#             var = mixture.var[j]
#             cov = np.diag(var[mask]) + np.eye(mask.sum()) * min_variance

#             # Compute log likelihood in the log domain
#             log_likelihoods[j] = multivariate_normal.logpdf(masked_X, mean=mean, cov=cov)
#             log_likelihoods[j] += np.log(mixture.p[j] + 1e-16)  # Add log prior probability

#         # Compute the log sum exp for normalization and update total log likelihood
#         logsumexp = scipy.special.logsumexp(log_likelihoods)
#         post[i, :] = np.exp(log_likelihoods - logsumexp)
#         total_log_likelihood += logsumexp

#     return post, total_log_likelihood


    # import matplotlib.pyplot as plt
    # from common import load_example
    # X = load_example()
    # mixture = GaussianMixture.multiple_init(X, 4)
    # mixture, post, ll = run(X, mixture, None)
    # X_pred = fill_matrix(X, mixture)
    # plt.subplot(1, 2, 1)
    # plt.imshow(X)
    # plt.title("Original")
    # plt.subplot(1, 2, 2)
    # plt.imshow(X_pred)
    # plt.title("Imputed")
    # plt.show()




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
#     n, d = X.shape
#     K = mixture.mu.shape[0]
#     new_mu = np.zeros((K, d))
#     new_p = np.zeros(K)
#     new_var = np.zeros(K)

#     # Create a mask for present values (non-zero)
#     present_mask = (X != 0)

#     for k in range(K):
#         # Compute log weights and stabilize
#         log_weights = np.log(post[:, k] + 1e-16)  # Log domain with small epsilon
#         max_log_weight = np.max(log_weights)
#         stabilized_log_weights = log_weights - max_log_weight

#         # Convert stabilized log weights back to standard domain
#         weights = np.exp(stabilized_log_weights)

#         # Normalize weights
#         weights /= weights.sum()

#         # Update the means
#         for feature in range(d):
#             mask = present_mask[:, feature]
#             weight_sum = weights[mask].sum()
#             if weight_sum > 0:
#                 new_mu[k, feature] = (X[mask, feature] @ weights[mask]) / weight_sum
#             else:
#                 new_mu[k, feature] = mixture.mu[k, feature]  # Default value handling

#         # Update the variances
#         squared_diff = (X - new_mu[k]) ** 2 * present_mask
#         weighted_squares = np.sum(weights[:, np.newaxis] * squared_diff, axis=0)
#         new_var[k] = np.maximum(min_variance, weighted_squares.sum() / (weights.sum() * np.sum(present_mask, axis=0).mean()))

#         # Update the mixing coefficients
#         new_p[k] = weights.mean()

#     return GaussianMixture(new_mu, new_var, new_p)

# below adds usage of min_variance

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
#     n, d = X.shape
#     K = mixture.mu.shape[0]
#     new_mu = np.zeros((K, d))
#     new_p = np.zeros(K)
#     new_var = np.zeros(K)

#     # Create a mask for present values (non-zero)
#     present_mask = (X != 0)

#     for k in range(K):
#         # Calculate the log weights for each data point for component k
#         log_weights = np.log(post[:, k] + 1e-16)  # Adding epsilon to avoid log(0)
#         max_log_weight = np.max(log_weights)
#         stabilized_log_weights = log_weights - max_log_weight

#         # Convert back to standard domain for mean calculation
#         weights = np.exp(stabilized_log_weights)

#         # Update the means
#         for feature in range(d):
#             mask = present_mask[:, feature]
#             weight_sum = weights[mask].sum()
#             if weight_sum > 0:
#                 new_mu[k, feature] = (X[mask, feature] @ weights[mask]) / weight_sum
#             else:
#                 # Handle the case where weight_sum is zero
#                 new_mu[k, feature] = mixture.mu[k, feature]  # or some other default value



#         # Update the variances
#         # Calculate the squared differences for present values only
#         squared_diff = (X - new_mu[k]) ** 2 * present_mask
#         weighted_squares = np.sum(weights[:, np.newaxis] * squared_diff, axis=0)
#         new_var[k] = np.maximum(min_variance, weighted_squares.sum() / (weights.sum() * np.sum(present_mask, axis=0).mean()))

#         # Update the mixing coefficients
#         new_p[k] = weights.mean()

#     return GaussianMixture(new_mu, new_var, new_p)





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
#     n, d = X.shape
#     K = mixture.mu.shape[0]
#     new_mu = np.zeros((K, d))
#     new_p = np.zeros(K)
#     new_var = np.zeros(K)

#     # Create a mask for present values (non-zero)
#     present_mask = (X != 0)

#     for k in range(K):
#         # Calculate the weights for each data point for component k
#         weights = post[:, k]

#         # Update the means
#         # Only use present values for each feature
#         for feature in range(d):
#             mask = present_mask[:, feature]
#             #new_mu[k, feature] = (X[mask, feature] @ weights[mask]) / weights[mask].sum()
#             weight_sum = weights[mask].sum()
#             if weight_sum > 0:
#                 new_mu[k, feature] = (X[mask, feature] @ weights[mask]) / weight_sum
#             else:
#                 # Handle the case where weight_sum is zero
#                 # For example, you might want to set it to the current mean or some default value
#                 new_mu[k, feature] = mixture.mu[k, feature]  # or some other default value


#         # Update the variances
#         # Calculate the squared differences for present values only
#         squared_diff = (X - new_mu[k]) ** 2 * present_mask
#         weighted_squares = np.sum(weights[:, np.newaxis] * squared_diff, axis=0)
#         new_var[k] = np.maximum(min_variance, weighted_squares.sum() / (weights.sum() * np.sum(present_mask, axis=0).mean()))

#         # Update the mixing coefficients
#         new_p[k] = weights.mean()

#     return GaussianMixture(new_mu, new_var, new_p)

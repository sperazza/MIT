import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer

def run_em_algorithm(data, K_values, seeds):
    # Use SimpleImputer to fill missing values with the mean
    imputer = SimpleImputer()
    data_imputed = imputer.fit_transform(data)

    best_log_likelihoods = {1: float("-inf"), 12: float("-inf")}

    for seed in seeds:
        for K in K_values:
            # Initialize Gaussian Mixture Model with multiple initializations
            gmm = GaussianMixture(n_components=K, random_state=seed, n_init=10)

            # Fit the model on the imputed data
            gmm.fit(data_imputed)

            # Calculate log likelihood
            log_likelihood = gmm.score(data_imputed)

            # Update the best log likelihood values
            if log_likelihood > best_log_likelihoods[K]:
                best_log_likelihoods[K] = log_likelihood

    return best_log_likelihoods

def main():
    # Load the incomplete data matrix
    data = np.loadtxt("netflix_incomplete.txt")

    # Specify values of K and seeds
    K_values = [1, 12]
    seeds = [0, 1, 2, 3, 4]

    # Run EM algorithm
    best_log_likelihoods = run_em_algorithm(data, K_values, seeds)

    # Report the results
    for K, log_likelihood in best_log_likelihoods.items():
        print(f"Log-likelihood K={K}: {log_likelihood}")

if __name__ == "__main__":
    main()
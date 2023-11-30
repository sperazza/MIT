import numpy as np
import em_matrix
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

# Initialize the mixture model and the soft counts
mixture, post = common.init(X, K, seed)

# Run the EM algorithm
final_mixture, final_post, log_likelihood = em_matrix.run(X, mixture, post)

# Fill in the missing values in X with the expected values from the final mixture model
X_pred = X.copy()
missing = (X == 0)
for i in range(K):
    # We fill in the missing values with the mean values of the corresponding Gaussian component
    X_pred[missing] = np.outer(final_post[:, i], final_mixture.mu[i])[missing]

# Calculate RMSE
rmse = np.sqrt(((X_pred - X_gold) ** 2).mean())

print(f"After a run\nMu:\n{final_mixture.mu}")
print(f"Var: {final_mixture.var}")
print(f"P: {final_mixture.p}")
print(f"post:\n{final_post}")
print(f"LL: {log_likelihood}")
print(f"X_gold:\n{X_gold}")
print(f"X_pred:\n{X_pred}")
print(f"RMSE: {rmse}")

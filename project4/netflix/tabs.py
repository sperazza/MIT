    for j in range(K):
        # Update means
        for feature in range(d):
            feature_present = (X[:, feature] != 0)
            if np.sum(feature_present) == 0:
                # If all values for this feature are missing, set mean to 0
                new_mu[j, feature] = 0
            else:
                numerator = np.sum(post[:, j] * feature_present * X[:, feature])
                denominator = np.sum(post[:, j] * feature_present)
                new_mu[j, feature] = numerator / denominator if denominator > 0 else 0

        nk_j = np.sum(post[:, j, np.newaxis] * (X != 0), axis=0)  # Sum of weights for non-missing features
        diff = np.where(X != 0, X - new_mu[j], 0)
        weighted_ssq_diff = np.sum(post[:, j, np.newaxis] * diff ** 2, axis=0)
        valid_features = nk_j > 0
        avg_var = np.where(valid_features, weighted_ssq_diff / nk_j, 0)
        new_var[j] = np.maximum(np.mean(avg_var[valid_features]), min_variance) if np.any(valid_features) else min_variance


# Update variances
        for j in range(K):
            total_variance = 0
            count = 0
            for feature in range(d):
                feature_present = (X[:, feature] != 0)
                if np.sum(feature_present) > 0:
                    diff = np.where(feature_present, X[:, feature] - new_mu[j, feature], 0)
                    squared_diff = diff ** 2
                    weighted_ssq_diff = np.sum(post[:, j] * squared_diff)
                    nk = np.sum(post[:, j] * feature_present)
                    variance = weighted_ssq_diff / nk
                    total_variance += np.maximum(variance, min_variance)
                    count += 1
            new_var[j] = total_variance / count if count > 0 else min_variance

        # Update variances
        for j in range(K):
            total_weighted_variance = 0
            total_weight = np.sum(post[:, j])  # Total weight for component j
            for feature in range(d):
                feature_present = (X[:, feature] != 0)
                if np.sum(feature_present) > 0:
                    diff = np.where(feature_present, X[:, feature] - new_mu[j, feature], 0)
                    squared_diff = diff ** 2
                    weighted_ssq_diff = np.sum(post[:, j] * squared_diff)
                    variance = weighted_ssq_diff / np.sum(post[:, j] * feature_present)
                    total_weighted_variance += np.maximum(variance, min_variance)
            new_var[j] = total_weighted_variance / total_weight if total_weight > 0 else min_variance




        # Update variances
        valid_diff = np.where(X != 0, X - new_mu[j], 0)
        squared_diff = (valid_diff ** 2) * post[:, j, np.newaxis]  # Adding new axis
        total_weight = np.sum(post[:, j, np.newaxis] * (X != 0), axis=0)  # Broadcasting fix
        avg_var = np.sum(squared_diff, axis=0) / np.maximum(total_weight, 1e-16)
        new_var[j] = np.maximum(np.mean(avg_var[total_weight > 1]), min_variance)  # thresholding variance update

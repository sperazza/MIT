import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    # Compute the dot product between each row of X and each row of Y
    dot_product = np.dot(X, Y.T)
    
    # Add the scalar c to each element of the resulting matrix
    kernel_matrix = dot_product + c
    
    # Raise each element of the matrix to the power p
    kernel_matrix = kernel_matrix ** p
    
    return kernel_matrix



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    # Calculate the squared Euclidean distance for every pair of rows
    sq_dist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(Y**2, 1) - 2 * np.dot(X, Y.T)
    
    # Compute the RBF kernel matrix
    kernel_matrix = np.exp(-gamma * sq_dist)
    
    return kernel_matrix
import numpy as np

def calculate_covariance_matrix(sigmas, corr_matrix):
    """
    Calculate the covariance matrix given asset volatilities and a correlation matrix.

    Parameters:
        sigmas (np.array): Array of asset volatilities.
        corr_matrix (np.array): Correlation matrix of assets.

    Returns:
        np.ndarray: Covariance matrix.
    """
    # Covariance is the product of volatilities and correlations
    return np.outer(sigmas, sigmas) * corr_matrix


def calculate_inverse_covariance(cov_matrix):
    """
    Computes the inverse of a covariance matrix.

    Parameters:
        cov_matrix (np.ndarray): Covariance matrix to invert.

    Returns:
        np.ndarray: Inverse covariance matrix.
    """
    # Inverse covariance matrix used for portfolio optimization
    return np.linalg.inv(cov_matrix)


def calculate_min_variance_weights(inv_cov_matrix):
    """
    Calculates portfolio weights for the minimum variance portfolio.

    Parameters:
        inv_cov_matrix (np.ndarray): Inverse of the covariance matrix.

    Returns:
        np.ndarray: Asset weights for the minimum variance portfolio.
    """
    one_vector = np.ones(inv_cov_matrix.shape[0])
    # Calculate weights ensuring total allocation sums to 1
    weights = np.dot(inv_cov_matrix, one_vector) / np.dot(one_vector.T, np.dot(inv_cov_matrix, one_vector))
    return weights


def calculate_target_return_weights(inv_cov_matrix, mus, target_return):
    """
    Computes portfolio weights that achieve a specified target return with minimum variance.

    Parameters:
        inv_cov_matrix (np.ndarray): Inverse covariance matrix.
        mus (np.ndarray): Expected returns vector.
        target_return (float): Desired target return.

    Returns:
        np.ndarray: Asset weights for the target return portfolio.
    """
    one_vector = np.ones(inv_cov_matrix.shape[0])

    # Compute intermediate variables for optimization equations
    A = np.dot(one_vector.T, np.dot(inv_cov_matrix, one_vector))
    B = np.dot(one_vector.T, np.dot(inv_cov_matrix, mus))
    C = np.dot(mus.T, np.dot(inv_cov_matrix, mus))

    # Lagrange multipliers for constrained optimization
    lambda_ = (C - B * target_return) / (A * C - B**2)
    gamma_ = (target_return * A - B) / (A * C - B**2)

    # Calculate final weights based on Lagrange multipliers
    weights = np.dot(inv_cov_matrix, lambda_ * one_vector + gamma_ * mus)
    return weights

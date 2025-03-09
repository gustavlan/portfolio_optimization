import numpy as np

def calculate_covariance_matrix(sigmas, corr_matrix):
    return np.outer(sigmas, sigmas) * corr_matrix

def calculate_inverse_covariance(cov_matrix):
    return np.linalg.inv(cov_matrix)

def calculate_min_variance_weights(inv_cov_matrix):
    one_vector = np.ones(inv_cov_matrix.shape[0])
    weights = np.dot(inv_cov_matrix, one_vector) / np.dot(one_vector.T, np.dot(inv_cov_matrix, one_vector))
    return weights

def calculate_target_return_weights(inv_cov_matrix, mus, target_return):
    one_vector = np.ones(inv_cov_matrix.shape[0])
    A = np.dot(one_vector.T, np.dot(inv_cov_matrix, one_vector))
    B = np.dot(one_vector.T, np.dot(inv_cov_matrix, mus))
    C = np.dot(mus.T, np.dot(inv_cov_matrix, mus))
    lambda_ = (C - B * target_return) / (A * C - B**2)
    gamma_ = (A * target_return - B) / (A * C - B**2)
    weights = np.dot(inv_cov_matrix, lambda_ * one_vector + gamma_ * mus)
    return weights

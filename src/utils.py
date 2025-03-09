import numpy as np

def transform_corr_matrix(corr_matrix, factor):
    """
    Transforms a correlation matrix by scaling it towards an identity matrix.

    Parameters:
        corr_matrix (np.ndarray): Original correlation matrix.
        factor (float): The scaling factor for transforming correlations.

    Returns:
        np.ndarray: Transformed correlation matrix.
    """
    identity_matrix = np.identity(len(corr_matrix))
    transformed_corr_matrix = corr_matrix * factor + identity_matrix * (1 - factor)
    return transformed_corr_matrix

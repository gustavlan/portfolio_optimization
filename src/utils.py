import numpy as np
from numpy.typing import NDArray

def transform_corr_matrix(corr_matrix: NDArray[np.float64], factor: float) -> NDArray[np.float64]:
    """
    Transforms a correlation matrix by scaling it towards (or beyond) the identity matrix.

    This function performs a linear interpolation (or extrapolation) between the original correlation matrix
    and the identity matrix. A factor of 1.0 returns the original matrix. Factors less than 1.0 reduce 
    the correlations (moving them toward 0), while factors greater than 1.0 amplify the correlations. 
    Regardless of the factor, the output values are clipped so that they remain within the valid range [-1, 1].

    Parameters:
        corr_matrix (np.ndarray): Original correlation matrix, expected to be a square matrix.
        factor (float): The scaling factor for transforming correlations. 
                        - A factor of 1.0 returns the original matrix.
                        - Factors < 1.0 reduce correlations.
                        - Factors > 1.0 amplify correlations.
                        Note: The transformed values are clipped to be within [-1, 1].

    Returns:
        np.ndarray: Transformed correlation matrix with all values in the range [-1, 1].

    Raises:
        ValueError: If 'corr_matrix' is not a square matrix.

    Examples:
        >>> import numpy as np
        >>> corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        >>> transform_corr_matrix(corr, 2.0)
        array([[1. , 1. ],
               [1. , 1. ]])
        >>> transform_corr_matrix(corr, 0.5)
        array([[1. , 0.4],
               [0.4, 1. ]])
    """
    # Validate that the correlation matrix is a square matrix.
    if corr_matrix.ndim != 2 or corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError("The 'corr_matrix' must be a square matrix.")

    identity_matrix = np.eye(corr_matrix.shape[0])
    # Compute the transformed matrix using linear interpolation/extrapolation.
    transformed_corr_matrix = corr_matrix * factor + identity_matrix * (1 - factor)
    # Clip the values to ensure they remain within the valid correlation range.
    transformed_corr_matrix = np.clip(transformed_corr_matrix, -1, 1)
    return transformed_corr_matrix

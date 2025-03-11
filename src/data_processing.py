import numpy as np
import pandas as pd

def create_asset_data(num_assets=4, mus=None, sigmas=None, generate_random=False, seed=None):
    """
    Creates asset data for portfolio optimization, expected returns and volatilities can be set 
    or generated randomly (within realistic ranges).

    Parameters:
        num_assets (int): Number of assets to include in the dataset.
        mus (list or np.array): List or array of expected returns. Must match num_assets if provided.
        sigmas (list or np.array): List or array of volatilities. Must match num_assets if provided.
        generate_random (bool): If True, generates random returns and volatilities with realistic relationships.
        seed (int, optional): Random seed for reproducibility of generated data.

    Returns:
        pd.DataFrame: DataFrame containing asset identifiers, expected returns, and volatilities.
        np.ndarray: Array of asset volatilities.
        np.ndarray: Array of asset expected returns.
    """

    if seed is not None:
        np.random.seed(seed)

    asset_labels = [f'Asset_{i+1}' for i in range(num_assets)]

    if generate_random:
        # Generate volatilities realistically between 5% and 40%
        sigmas = np.random.uniform(0.05, 0.4, num_assets)
        # Generate returns with realistic risk-return trade-off, roughly proportional to volatility
        mus = np.random.uniform(0.03, 0.15, num_assets) * (sigmas / np.mean(sigmas))

    else:
        if mus is None or sigmas is None:
            raise ValueError("When 'generate_random' is False, both 'mus' and 'sigmas' must be provided.")
        if len(mus) != num_assets or len(sigmas) != num_assets:
            raise ValueError("Length of 'mus' and 'sigmas' must match 'num_assets'.")

        mus = np.array(mus)
        sigmas = np.array(sigmas)

    assets_df = pd.DataFrame({
        'asset': asset_labels,
        'mu': mus,
        'sigma': sigmas
    })

    return assets_df, sigmas, mus

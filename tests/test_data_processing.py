import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import create_asset_data

def test_manual_asset_input():
    """
    Tests that create_asset_data works correctly when given manual asset expected returns and volatilities.
    """
    num_assets = 4
    mus = [0.05, 0.07, 0.12, 0.03]
    sigmas = [0.07, 0.28, 0.35, 0.18]
    assets_df, sigmas_out, mus_out = create_asset_data(num_assets=num_assets, mus=mus, sigmas=sigmas, generate_random=False)
    assert len(assets_df) == num_assets
    assert all(sigmas_out == sigmas)
    assert all(mus_out == mus)

def test_random_asset_generation():
    """
    Tests that create_asset_data generates random data correctly when given a seed.
    """
    assets_df, sigmas, mus = create_asset_data(num_assets=5, generate_random=True, seed=42)
    assert len(assets_df) == 5
    assert not assets_df.isnull().values.any()

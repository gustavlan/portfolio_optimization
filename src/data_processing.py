import numpy as np
import pandas as pd

def create_asset_data():
    assets_list = [['A', 0.05, 0.07],
                   ['B', 0.07, 0.28],
                   ['C', 0.12, 0.35],
                   ['D', 0.03, 0.18]]
    assets = pd.DataFrame(assets_list, columns=['asset', 'mu', 'sigma'])
    sigmas = assets['sigma'].values
    mus = assets['mu'].values
    return assets, sigmas, mus

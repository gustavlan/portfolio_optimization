# Portfolio Optimization

This is a quantitative finance project that implements advanced asset allocation techniques using Python. Focus is on highlighting mathematical derivations leading to practical applications of quantative methods.

## Overview

This repository provides a comprehensive framework for optimizing investment portfolios. Key features include:

- **Mathematical Derivations:** Detailed derivations of portfolio optimization formulas are presented in the interactive Jupyter Notebook.
- **Functions:** Functions to compute covariance matrices, inverse covariance matrices, and optimal portfolio weights—both for the minimum variance portfolio and portfolios targeting a specific return.
- **Data Processing:** Robust tools to generate and process realistic asset data, simulating real-world market conditions.
- **Utility Functions:** Additional tools such as correlation matrix transformations to adjust risk metrics.

## Methodologies & Mathematical Derivations

### Covariance Matrix Calculation

The covariance matrix \(\Sigma\) is computed using asset volatilities and pairwise correlations:
  
\[
\Sigma_{ij} = \sigma_i \sigma_j \rho_{ij}
\]

This formulation is the  quantification of risk in this portfolio optimization.

### Minimum Variance Portfolio

The optimal weights \(w\) for the minimum variance portfolio are derived by:

\[
w = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^T \Sigma^{-1} \mathbf{1}}
\]

where \(\mathbf{1}\) is a vector of ones and \(\Sigma^{-1}\) is the inverse covariance matrix. This approach minimizes the portfolio variance subject to the investment constraint.

### Target Return Portfolio Optimization

For portfolios aiming at a specific target return \(R_t\), weights are computed using Lagrange multipliers. Define:

\[
A = \mathbf{1}^T \Sigma^{-1} \mathbf{1}, \quad B = \mathbf{1}^T \Sigma^{-1} \mu, \quad C = \mu^T \Sigma^{-1} \mu
\]

Then, the Lagrange multipliers are given by:

\[
\lambda = \frac{C - B R_t}{AC - B^2}, \quad \gamma = \frac{R_t A - B}{AC - B^2}
\]

and the optimal weights are:

\[
w = \Sigma^{-1} \left( \lambda \mathbf{1} + \gamma \mu \right)
\]

These derivations are detailed in the [Jupyter Notebook](./notebooks/portfolio_optimization.ipynb) provided with this repository.

## Project Structure

```
portfolio_optimization/
├── README.md
├── setup.py
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── portfolio_optimizer.py   # Core portfolio optimization functions
│   ├── utils.py                 # Utility functions (e.g., correlation matrix transformation)
│   └── data_processing.py       # Functions for generating and processing asset data
└── notebooks/
    └── portfolio_optimization.ipynb   # Interactive Jupyter Notebook for analysis
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/portfolio_optimization.git
   cd portfolio_optimization
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package:**

   ```bash
   pip install .
   ```

## Expanded Usage

This section provides comprehensive examples on how to leverage the functionalities in the project.

### 1. Data Processing

You can generate asset data with either manually provided expected returns and volatilities or generate them randomly:

```python
from src.data_processing import create_asset_data

# Example 1: Generate random asset data for 5 assets
assets_df, sigmas, mus = create_asset_data(num_assets=5, generate_random=True, seed=42)
print("Generated Asset Data:")
print(assets_df)

# Example 2: Generate asset data with custom parameters
custom_mus = [0.1, 0.12, 0.08, 0.11]
custom_sigmas = [0.2, 0.25, 0.18, 0.22]
assets_df_custom, sigmas_custom, mus_custom = create_asset_data(num_assets=4, mus=custom_mus, sigmas=custom_sigmas)
print("Custom Asset Data:")
print(assets_df_custom)
```

### 2. Portfolio Optimization

#### 2.1 Minimum Variance Portfolio

Use the core functions to compute the minimum variance portfolio:

```python
from src.portfolio_optimizer import calculate_covariance_matrix, calculate_inverse_covariance, calculate_min_variance_weights
import numpy as np

# Create a simple correlation matrix for the assets
# For demonstration, assume a constant correlation of 0.5 between different assets
num_assets = len(sigmas)
corr_matrix = np.full((num_assets, num_assets), 0.5)
np.fill_diagonal(corr_matrix, 1.0)

# Calculate the covariance matrix
cov_matrix = calculate_covariance_matrix(sigmas, corr_matrix)
print("Covariance Matrix:")
print(cov_matrix)

# Compute the inverse covariance matrix and the optimal weights
inv_cov_matrix = calculate_inverse_covariance(cov_matrix)
min_var_weights = calculate_min_variance_weights(inv_cov_matrix)
print("Minimum Variance Portfolio Weights:")
print(min_var_weights)
```

#### 2.2 Target Return Portfolio

Calculate portfolio weights that aim to achieve a specific target return:

```python
from src.portfolio_optimizer import calculate_target_return_weights

# Define a target return (e.g., 10%)
target_return = 0.10

# Compute the portfolio weights along with the portfolio's expected return and volatility
target_weights, portfolio_mu, portfolio_sigma = calculate_target_return_weights(cov_matrix, mus, target_return)
print("Target Return Portfolio Weights:")
print(target_weights)
print(f"Portfolio Expected Return: {portfolio_mu:.4f}")
print(f"Portfolio Volatility: {portfolio_sigma:.4f}")
```

### 3. Correlation Matrix Transformation

Utilize the utility function to adjust the correlation matrix by scaling it toward or away from the identity matrix:

```python
from src.utils import transform_corr_matrix

# Example: Amplify correlations by a factor of 1.2
amplified_corr = transform_corr_matrix(corr_matrix, factor=1.2)
print("Amplified Correlation Matrix:")
print(amplified_corr)

# Example: Dampen correlations by a factor of 0.8
dampened_corr = transform_corr_matrix(corr_matrix, factor=0.8)
print("Dampened Correlation Matrix:")
print(dampened_corr)
```

### 4. Running the Jupyter Notebook

The interactive Jupyter Notebook provides a detailed walk-through of the entire process—from data generation to portfolio optimization. To launch the notebook:

```bash
jupyter notebook notebooks/portfolio_optimization.ipynb
```

The notebook includes:
- Step-by-step explanations of the mathematical derivations.
- Interactive visualizations of portfolio risk-return trade-offs.
- Detailed examples of sensitivity analysis and scenario testing.

## Conclusion

This project exemplifies a robust and mathematically sound approach to portfolio optimization, blending theory with practical implementation. By exploring the examples provided above and the interactive Jupyter Notebook, users can gain a deep understanding of quantitative finance methods.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

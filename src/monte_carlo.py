from numba import njit, prange
import numpy as np

# Define kernel for GPU processing
@njit(parallel=True)
def simulate_portfolios(
    number_of_simulations,
    number_of_stocks,
    mean_table,
    covariance_table,
    risk_free_rate,
    all_weights,
    returns_array,
    volatilities_array,
    sharpe_array
):
    for i in prange(number_of_simulations):
        # Generate random weights for the portfolio
        weights = np.random.random(number_of_stocks)
        weights /= np.sum(weights)  # Normalize weights
        all_weights[i, :] = weights

        # Calculate expected return
        returns_array[i] = np.sum(mean_table * weights)

        # Calculate expected volatility
        volatilities_array[i] = np.sqrt(np.dot(weights.T, np.dot(covariance_table, weights)))

        # Calculate Sharpe ratio
        sharpe_array[i] = (returns_array[i] - risk_free_rate) / volatilities_array[i]
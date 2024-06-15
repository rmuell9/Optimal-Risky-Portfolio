from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

tickers = []
timeScale = 5
maxWeight = 0.5
print(
    "Enter the tickers to include in an optimal risky portfolio. Enter nothing to exit."
)
while True:
    i = input()
    if i == "":
        break
    else:
        tickers.append(i)
print(tickers)
print(
    "Enter the time frame for the returns (yrs). The default is 5 yrs. "
)
while True:
    i = input()
    if i == "":
        break
    else:
        timeScale = float(i)
        break

print("Enter the maximum asset weight in decimal form (.5 of portfolio is the default max)")
while True:
    i = input()
    if i == "":
        break
    else:
        maxWeight = float(i)
        break

endDate = datetime.today()

startDate = endDate - timedelta(days=365 * timeScale)

adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start=startDate, end=endDate)
    adj_close_df[ticker] = data["Adj Close"]

log_returns = np.log(adj_close_df / adj_close_df.shift(1))

log_returns = log_returns.dropna()

cov_matrix = log_returns.cov() * 252

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252

def sharpe_ratio(weights, log_returns, cov_matrix, rf_rate):
    return (expected_return(weights, log_returns) - rf_rate) / standard_deviation(
        weights, cov_matrix
    )

rf_rate = 0.044

def neg_sharpe_ratio(weights, log_returns, cov_matrix, rf_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, rf_rate)

constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
bounds = [(0, maxWeight) for _ in range(len(tickers))]

initial_weights = np.array([1 / len(tickers)] * len(tickers))

optimized_results = minimize(
    neg_sharpe_ratio,
    initial_weights,
    args=(log_returns, cov_matrix, rf_rate),
    method="SLSQP",
    constraints=constraints,
    bounds=bounds,
)

optimal_weights = optimized_results.x

print("Optimal Portfolio Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")
print()

eRP = expected_return(optimal_weights, log_returns)
vol = standard_deviation(optimal_weights, cov_matrix)
sharpe = sharpe_ratio(optimal_weights, log_returns, cov_matrix, rf_rate)

print(f"Expected Annual Return: {eRP:.4f}")
print(f"Expected Volatility: {vol:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(tickers, optimal_weights)

plt.xlabel("Assets")
plt.ylabel("Optimal Weights")
plt.title("Optimal Portfolio Weights")

plt.show()

from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import re

timeScale = 5
maxWeight = 0.5

def clean_tickers(ticker_list):
    cleaned_tickers = [ticker.replace('.', '-') for ticker in ticker_list if isinstance(ticker, str) and re.match("^[A-Z0-9]+$", ticker)]
    return cleaned_tickers

# Fetch the list of S&P 500 companies
NASDAQ = pd.read_csv('/Users/matthewmueller/Desktop/nasdaq_screener_1726641015074.csv')
tickers = NASDAQ['Symbol'].tolist()
tickers = clean_tickers(tickers)
tickers = tickers[:4000]

print("Enter the time frame for the returns (yrs). The default is 5 yrs.")
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

print("Enter the current risk-free rate (e.g., 0.05 for 5%). The default is 0.044.")
while True:
    i = input()
    if i == "":
        rf_rate = 0.044
        break
    else:
        rf_rate = float(i)
        break

endDate = datetime.today()
startDate = endDate - timedelta(days=365 * timeScale)

# Download data for all tickers at once with adjusted close prices
print("Downloading data for all tickers...")
data = yf.download(tickers, start=startDate, end=endDate, auto_adjust=True)

# Inspect data structure
print("Data Columns:")
print(data.columns)
print("\nData Head:")
print(data.head())

# Extract adjusted 'Close' prices
try:
    adj_close_df = data['Close']
except KeyError:
    adj_close_df = data

# Check if 'adj_close_df' is empty
if adj_close_df.empty:
    raise ValueError("Adjusted Close data is empty. Please check the ticker symbols and data availability.")

# Drop columns (tickers) with any missing data
adj_close_df = adj_close_df.dropna(axis=1, how='any')

# Update ticker list to include only tickers with complete data
tickers = adj_close_df.columns.tolist()

# Check if we have enough tickers after cleaning
if len(tickers) == 0:
    raise ValueError("No tickers have complete data. Cannot proceed.")

# Compute arithmetic returns
returns = adj_close_df.pct_change().dropna()

# Annualize covariance matrix
cov_matrix = returns.cov() * 252

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, returns):
    return np.sum(returns.mean() * weights) * 252

def sharpe_ratio(weights, returns, cov_matrix, rf_rate):
    return (expected_return(weights, returns) - rf_rate) / standard_deviation(weights, cov_matrix)

def neg_sharpe_ratio(weights, returns, cov_matrix, rf_rate):
    return -sharpe_ratio(weights, returns, cov_matrix, rf_rate)

constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
bounds = [(0, maxWeight) for _ in range(len(tickers))]
initial_weights = np.array([1 / len(tickers)] * len(tickers))

print("Optimizing portfolio... This may take a while.")
optimized_results = minimize(
    neg_sharpe_ratio,
    initial_weights,
    args=(returns, cov_matrix, rf_rate),
    method="SLSQP",
    constraints=constraints,
    bounds=bounds,
    options={'maxiter': 1000, 'disp': True}
)

if not optimized_results.success:
    print("Optimization failed:", optimized_results.message)
    exit()

optimal_weights = optimized_results.x

print("\nOptimal Portfolio Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")
print()

eRP = expected_return(optimal_weights, returns)
vol = standard_deviation(optimal_weights, cov_matrix)
sharpe = sharpe_ratio(optimal_weights, returns, cov_matrix, rf_rate)

print(f"Expected Annual Return: {eRP:.4f}")
print(f"Expected Volatility: {vol:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")

# Plot significant weights (e.g., weights > 1%)
significant_weights = {ticker: weight for ticker, weight in zip(tickers, optimal_weights) if weight > 0.01}

if significant_weights:
    plt.figure(figsize=(12, 6))
    plt.bar(significant_weights.keys(), significant_weights.values())
    plt.xlabel("Assets")
    plt.ylabel("Optimal Weights")
    plt.title("Optimal Portfolio Weights (Weights > 1%)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
else:
    print("No weights greater than 1% to display.")

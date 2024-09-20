import yfinance as yf, pandas as pd, pandas_market_calendars as mcal, numpy as np, re, matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import f
from datetime import datetime, timedelta
from scipy.optimize import minimize

nyse = mcal.get_calendar('NYSE')
today = pd.Timestamp.today()
start_date = today - pd.Timedelta(days=15)
schedule = nyse.schedule(start_date = start_date, end_date = today)
last_10_trading_days = schedule.index[-11:]

def clean_tickers(ticker_list):
    cleaned_tickers = [ticker.replace('.', '-') for ticker in ticker_list if isinstance(ticker, str) and re.match("^[A-Z0-9]+$", ticker)]
    return cleaned_tickers

benchmark = 'SPY'

SPY = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(SPY)
SPY_table = tables[0]
tickers = SPY_table['Symbol'].tolist()
tickers = [ticker.replace('.', '-') for ticker in tickers]

NASDAQ = pd.read_csv('/Users/matthewmueller/Desktop/nasdaq_screener_1726641015074.csv')
ntickers = NASDAQ['Symbol'].tolist()
ntickers = clean_tickers(ntickers)
ntickers = ntickers[:4000]

good_returns = []
good_risk = []

def getReturns(benchmark, tickers):
    bm_data = yf.download(benchmark, start = last_10_trading_days[0], end = today)
    bm_adj_close = bm_data['Adj Close']
    bm_returns = bm_adj_close.pct_change().dropna() * 100
    X = bm_returns.values
    XMean = np.mean(X)
    XVar = np.var(X, ddof=1)
    
    data = yf.download(tickers, start=last_10_trading_days[0], end=today, group_by='ticker', threads=True)

    alpha = 0.05

    for ticker in tickers:
        try:
            adj_close = data[ticker]['Adj Close']
            returns = adj_close.pct_change().dropna() * 100
            Y = returns.values
            YMean = np.mean(Y)
            YVar = np.var(Y, ddof=1)

            t_stat, p_val = stats.ttest_ind(X, Y, equal_var=False)
            if (p_val < alpha) and (YMean > XMean):
                good_returns.append(ticker)
            
            f_stat = np.var(X, ddof=1) / np.var(Y, ddof=1)
            dof1 = len(X) - 1 
            dof2 = len(Y) - 1
            p_value = 2 * min(f.cdf(f_stat, dof1, dof2), 1 - f.cdf(f_stat, dof1, dof2))
            if (p_value < alpha) and (YVar < XVar):
                good_risk.append(ticker)
            if (ticker in good_returns) and (p_value < alpha) and (YVar > XVar):
                good_returns.remove(ticker)
        
        except Exception as e: 
            print(f'Error for {ticker}: {e}')

getReturns(benchmark, ntickers)

golden = list(set(good_returns).intersection(good_risk))

print(f'Good returns: {good_returns}')

print(f'Golden: {golden}')

def portfolioOptimize(tickers): 
    timeScale = 5
    maxWeight = 1
    rf_rate = 0.044

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

portfolioOptimize(good_returns)
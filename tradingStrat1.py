import yfinance as yf, pandas as pd, pandas_market_calendars as mcal, numpy as np

from scipy import stats
from scipy.stats import f

nyse = mcal.get_calendar('NYSE')
today = pd.Timestamp.today()
start_date = today - pd.Timedelta(days=15)
schedule = nyse.schedule(start_date = start_date, end_date = today)
last_10_trading_days = schedule.index[-11:]

benchmark = 'SPY'

SPY = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(SPY)
SPY_table = tables[0]
tickers = SPY_table['Symbol'].tolist()

good_returns = []
good_risk = []

def getReturns(benchmark, tickers):
    bm_adj_close = pd.DataFrame()
    bm_data = yf.download(benchmark, start = last_10_trading_days[0], end = today)
    bm_adj_close[benchmark] = bm_data['Adj Close']
    bm_returns = ((bm_adj_close / bm_adj_close.shift(1)) -1) * 100
    bm_returns = bm_returns.dropna()
    A = tuple(bm_returns[benchmark].round(6))
    X = np.array(A)
    XMean = np.mean(X)
    XVar = np.var(X, ddof=1)
    
    for ticker in tickers:
        adj_close_df = pd.DataFrame()
        data = yf.download(ticker, start = last_10_trading_days[0], end = today)
        adj_close_df[ticker] = data['Adj Close']

        returns = ((adj_close_df / adj_close_df.shift(1)) -1) * 100
        returns = returns.dropna()
        B = tuple(returns[ticker].round(6))
        Y = np.array(B)
        YMean = np.mean(Y)
        YVar = np.var(Y, ddof=1)
        t_stat, p_val = stats.ttest_ind(X, Y, equal_var=False)
        if (p_val < 0.05) & (YMean > XMean):
            good_returns.append(ticker)
        f_stat = np.var(X, ddof=1) / np.var(Y, ddof=1)
        dof1 = len(X) - 1  # degrees of freedom for X
        dof2 = len(Y) - 1  # degrees of freedom for Y
        p_value = 2 * min(f.cdf(f_stat, dof1, dof2), 1 - f.cdf(f_stat, dof1, dof2))
        if (p_value < 0.05) & (YVar < XVar):
            good_risk.append(ticker)
        


getReturns(benchmark, tickers)

print(f'Good returns: {good_returns}')
print(f'Good risk: {good_risk}')
        



    

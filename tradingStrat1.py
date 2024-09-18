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
tickers = [ticker.replace('.', '-') for ticker in tickers]

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
        
        except Exception as e: 
            print(f'Error for {ticker}: {e}')

getReturns(benchmark, tickers)

print(f'Good returns: {good_returns}')
print(f'Good risk: {good_risk}')


        



    

import yfinance as yf, pandas as pd, pandas_market_calendars as mcal

nyse = mcal.get_calendar('NYSE')
benchmark = 'SPY'

today = pd.Timestamp.today()
start_date = today - pd.Timedelta(days=15)

schedule = nyse.schedule(start_date = start_date, end_date = today)

last_10_trading_days = schedule.index[-11:]

print(last_10_trading_days)

adj_close_df = pd.DataFrame()
data = yf.download(benchmark, start = last_10_trading_days[0], end = today)
adj_close_df[benchmark] = data['Adj Close']

print(adj_close_df)


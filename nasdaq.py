import pandas as pd, re

def clean_tickers(ticker_list):
    cleaned_tickers = [ticker for ticker in ticker_list if '^' not in ticker]
    return cleaned_tickers

NASDAQ = pd.read_csv('/Users/matthewmueller/Desktop/nasdaq_screener_1726641015074.csv')
ntickers = NASDAQ['Symbol'].tolist()
ntickers = clean_tickers(ntickers)
print(ntickers)
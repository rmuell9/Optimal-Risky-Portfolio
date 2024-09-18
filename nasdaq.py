import pandas as pd

NASDAQ = pd.read_csv('/Users/matthewmueller/Desktop/nasdaq_screener_1726641015074.csv')
ntickers = NASDAQ['Symbol'].tolist()
print(ntickers)
import pandas as pd

SPY = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

tables = pd.read_html(SPY)

SPY_table = tables[0]

tickers = SPY_table['Symbol'].tolist()

print(tickers)

import yfinance as yf
import os
import json
import pandas as pd

API_PATH = "msft_data.json"

if os.path.exists(API_PATH):
    #If data exists locally we use that instead of redownloading and getting potentially limited
    with open(API_PATH) as f:
        msft_prices = pd.read_json(API_PATH)
else:
    #Pull data from yfiance and convert it into a JSON file
    msft = yf.Ticker("MSFT")
    msft_prices = msft.history(period="max")

    msft_prices.to_json(API_PATH)

print(msft_prices.plot.line(y="Close", use_index=True))
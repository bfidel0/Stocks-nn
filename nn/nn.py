import yfinance as yf
import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import numpy as np

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

#print(msft_prices.plot.line(y="Close", use_index=True))

#Determines the actual closing prices
data = msft_prices[["Close"]]
data = data.rename(columns= {"Close": "Actual_Close"})


# sets our target price indicating if price went up or down
data["Target"] = msft_prices.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]


#shift the stock price forward one day 
msft_prev = msft_prices.copy()
msft_prev = msft_prev.shift(1)

#training data
predictors = ["Close", "Volume", "Open", 'High', "Low"]
data = data.join(msft_prev[predictors]).iloc[1:]

#print(data.head(5))


'''
The machine learning model starts here
'''
#The min sample split ensures we dont overfit the data
model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)

training_set = data.iloc[:-100]
test_set = data.iloc[-100:]

model.fit(training_set[predictors], training_set["Target"])


#Evaluates the error in our predictions
preds = model.predict(test_set[predictors])
preds = pd.Series(preds, index=test_set.index)


'''
Printing error test is ~51% which meansd we are barely better than coin flip
'''
#print(precision_score(test_set["Target"], preds))


def backtest(data, model, predictors, start=1000, step=750):
    predictons = []

    for i in range(start, data.shape[0], step):
        training_set = data.iloc[0:i].copy()
        test_set = data.iloc[i:(i+step)].copy()

        model.fit(training_set[predictors], training_set["Target"])

        preds = model.predict_proba(test_set[predictors])[:,-1]
        preds = pd.Series(preds, index=test_set.index)
        preds[preds > .6] = 1
        preds[preds <= .6] = 0

        combined = pd.concat({"Target": test_set["Target"], "Predictions": preds}, axis=1)

        predictons.append(combined)

    return pd.concat(predictons)


#predictions = backtest(data, model, predictors)

#print(predictions["Predictions"].value_counts())



#Improving accuracy functions

weekly_mean = data.rolling(7).mean()
quarterly_mean = data.rolling(90).mean()
annual_mean = data.rolling(365).mean()
weekly_trend = data.shift(1).rolling(7).mean()["Target"]

data["weekly_mean"] = weekly_mean["Close"] / data["Close"]
data["quarterly_mean"] = quarterly_mean["Close"] / data["Close"]
data["annual_mean"] = annual_mean["Close"] / data["Close"]

data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
data["weekly_trend"] = weekly_trend

data["open_close_ratio"] = data["Open"] / data["Close"]
data["high_close_ratio"] = data["High"] / data["Close"]
data["low_close_ratio"] = data["Low"] / data["Close"]

full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_close_ratio", "low_close_ratio", "weekly_trend"]


predictions = backtest(data.iloc[365:], model, full_predictors)

#print(predictions)

#Shows accuracy of model
print(precision_score(predictions["Target"], predictions["Predictions"]))

#shows how many tardes we would make
print(predictions["Predictions"].value_counts())
import yfinance as yf
import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
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


#Determines the actual closing prices for future use
data = msft_prices[["Close"]]
data = data.rename(columns= {"Close": "Actual_Close"})


'''
indicates whether the price of the stock went up or down. We use the rolling method to view today
vs the next day and that will return 1 if the price went up or 0 if the price went down.
'''
data["Target"] = msft_prices.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
msft_prev = msft_prices.copy()
msft_prev = msft_prev.shift(1)

#training data
predictors = ["Close", "Volume", "Open", 'High', "Low"]
data = data.join(msft_prev[predictors]).iloc[1:]

'''

This area is designated for GridsearchCV to determine the optimal parameters 
for our model. 

'''
'''

params_to_test = {
    'n_estimators': [100, 40, 50, 60, 75,],
    'max_depth' : [3,5,6]
}
model = RandomForestClassifier(random_state=42)
training_set = data.iloc[:-100]
test_set = data.iloc[-100:]
grid_search = GridSearchCV(model, param_grid=params_to_test, scoring='accuracy')

grid_search.fit(training_set[predictors], training_set["Target"])

best_params = grid_search.best_params_ 

#best_params is a dict you can pass directly to train a model with optimal settings 
best_model = grid_search.best_estimator_


'''

'''
The machine learning model starts here. We use a classfier since we are determining a binary decision
'''
model = RandomForestClassifier(n_estimators=300, min_samples_split=200, random_state=1)

training_set = data.iloc[:-100]
test_set = data.iloc[-100:]

model.fit(training_set[predictors], training_set["Target"])


#Evaluates the error in our predictions.
#Turns the numbpy array from the predict method into a pandas series
preds = model.predict(test_set[predictors])
preds = pd.Series(preds, index=test_set.index)


'''
Printing error test is ~51% which meansd we are barely better than coin flip
'''
#print(precision_score(test_set["Target"], preds))

'''
The backtest function is used to train the model every 50 rows as oppopssed to just training it once

A Step size of 50 takes a long time but pushes us >60% precision score
'''
def backtest(data, model, predictors, start=2000, step=50):
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



'''
We can add more prediction methods to increase the accuracy of our model
So we add rolling means to help predict upward or downward price movement aka volitility
of future days

'''

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

#Shows accuracy of model
print(precision_score(predictions["Target"], predictions["Predictions"]))


#shows how many trades we would make
print(predictions["Predictions"].value_counts())


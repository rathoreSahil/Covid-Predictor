#Sahil Singh Rathore
#B20227
#Mobile No: 9559176048

import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from numpy.lib import corrcoef
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math

df = pd.read_csv("C:\python\ds3_assn6\daily_covid_cases.csv",sep = ',')

#Question 1
df['Date'] = pd.to_datetime(df['Date'])
plt.plot(df['Date'],df['new_cases'])
plt.xlabel("Date")
plt.ylabel("New Confirmed Cases")
plt.xticks(rotation='vertical')
plt.show()

x = df['new_cases']
autocorr = np.corrcoef(x[:-1],x[1:])
print(autocorr[0][1])

plt.scatter(x[:-1],x[1:],s=5)
plt.xlabel("Original Sequence")
plt.ylabel("Lag Sequence")
plt.show()

corr = list()
for p in range(1,7):
    corr.append(np.corrcoef(x[:-p],x[p:])[0][1])

print(corr)
plt.plot(np.arange(1,7),corr)
plt.xlabel("Lag Values")
plt.ylabel("Corr Coef")
plt.show()

sm.graphics.tsa.plot_acf(x,lags = np.arange(1,25))
plt.xlabel("Lags")
plt.ylabel("AutoCorrelation")
plt.show()

# QUESTION 2
from statsmodels.tsa.ar_model import AutoReg as AR
from sklearn.metrics import mean_squared_error

# MAPE Function
def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

series = pd.read_csv('C:\python\ds3_assn6\daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

#AutoRegression Model
def AutoRegression(p):
    window = p
    model = AR(train, lags=window)
    model_fit = model.fit()
    coef = model_fit.params
    # walk forward over time steps in test
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
    return predictions

predictions = AutoRegression(5)
rmse = math.sqrt(mean_squared_error(test, predictions))*100/mean(test)
print("mean of test = ",mean(test))
mape = MAPE(test,predictions)
print('Test RMSE for p = {} is {r:.3f}'.format(5,r=rmse))
print('Test MAPE for p = {} is {m:.3f}'.format(5,m=mape))
plt.scatter(test, predictions)
plt.xlabel("actual")
plt.ylabel("Predicted")
plt.show()
plt.plot(np.arange(len(test)),test,label = 'actual')
plt.plot(np.arange(len(predictions)),predictions,color = 'red',label = 'predicted')
plt.legend()
plt.show()

# QUESTION 3
rmse = []
mape = []
lags = [1,5,10,15,25]
for p in lags:
    predictions = AutoRegression(p)
    rmse.append(math.sqrt(mean_squared_error(test, predictions))*100/mean(test))
    mape.append(MAPE(test,predictions))
print(rmse)
print(mape)

plt.bar([str(i) for i in lags],rmse,color = 'red')
plt.xlabel("Lag Value")
plt.xticks([str(i) for i in lags])
plt.ylabel("RMSE")
plt.title("RMSE")
plt.show()
plt.bar([str(i) for i in lags],mape,color = 'green')
plt.xlabel("Lag Value")
plt.ylabel("MAPE")
plt.xticks([str(i) for i in lags])
plt.title("MAPE")
plt.show()

#QUESTION 4
autocorr,p = 1,1
T = len(train)
while(abs(autocorr)>(2/np.sqrt(T))):
    autocorr = np.corrcoef(x[:-p],x[p:])[0][1]
    p+=1
heuristic = p-2
print("Heuristic Value = {}".format(heuristic))
predictions = AutoRegression(heuristic)
rmse=math.sqrt(mean_squared_error(test, predictions))
mape=MAPE(test,predictions)
print('Test RMSE for p = {} is {r:.3f}'.format(heuristic,r=rmse))
print('Test MAPE for p = {} is {m:.3f}'.format(heuristic,m=mape))

model = AR(X, lags=heuristic)
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
history = test[len(test)-heuristic:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(121):
    length = len(history)
    lag = [history[i] for i in range(length-heuristic, length)]
    yhat = coef[0]
    for d in range(heuristic):
        yhat += coef[d+1] * lag[heuristic-d-1]
    history.append(yhat)
    predictions.append(yhat)
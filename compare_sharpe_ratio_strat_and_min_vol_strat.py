import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pandas_datareader.data as pdrd
import scipy.optimize as solver
from functools import reduce
from datetime import datetime
from typing import List
# from utility_functions import *

%run utility_functions.ipynb

default_data_source = 'yahoo'
default_risk_free_rate = 0.02
window = 252
first_trading_date = datetime(2016, 1, 1)

#collect data from Yahoo
df_prices = collectdata(tickers=['GOOG', 'MSFT', 'AMZN', 'AAPL', 'NFLX'], start=datetime(2015,1,1), end=datetime(2019,12,31), data_source = default_data_source)

#get returns for the tickers
df_return = getreturns(df_prices)

# Determine the first trading day of every month
first_tradeday_month = df_prices.groupby(pd.Grouper(key = 'Date', freq = 'M')).Date.first()
rebalance_tradeday_monthly = first_tradeday_month[first_tradeday_month > first_trading_date]
rebalance_tradeday_monthly_index = df_prices[df_prices.Date.isin(rebalance_tradeday_monthly)].index.tolist() #list
opt_max_sharpe_ratio = []
opt_max_sharpe_weight = []

# print(rebalance_tradeday_monthly_index)

for i in range(len(rebalance_tradeday_monthly_index)):
    df_oneyear_return = df_return.iloc[rebalance_tradeday_monthly_index[i]-252:rebalance_tradeday_monthly_index[i]] #one year
    ratio, weight = max_sharpe_ratio_strat(df_oneyear_return = df_oneyear_return, risk_free_rate = default_risk_free_rate)
    weight_1 = np.around(weight, decimals = 4)
    opt_max_sharpe_ratio.append(ratio)
    opt_max_sharpe_weight.append(weight_1)
    
##maximum Sharpe Ratio Strategy  

df_opt_max_sharpe = pd.DataFrame(list(zip(opt_max_sharpe_ratio,opt_max_sharpe_weight)), columns = ['Sharpe Ratio', 'Weight ratio'])
print('The data frame of maximum Sharpe Ratio strategy: \n', df_opt_max_sharpe)

##minimum volatility strategy
opt_min_vol_return = []
opt_min_vol = []
opt_min_vol_weight = []
for i in range(len(rebalance_tradeday_monthly_index)):
    df_oneyear_return = df_return.iloc[rebalance_tradeday_monthly_index[i]-252:rebalance_tradeday_monthly_index[i]] #one year
    min_vol_return, min_vol_result, min_vol_weight = min_vol_strat(df_oneyear_return = df_oneyear_return)
    opt_min_vol_return.append(min_vol_return)
    opt_min_vol.append(min_vol_result)
    opt_min_vol_weight.append(np.around(min_vol_weight, decimals=4))

df_opt_min_vol_result = pd.DataFrame(list(zip(opt_min_vol_return, opt_min_vol, opt_min_vol_weight)), columns = ['return', 'minimum volatility', 'weight ratio'])
print('The data frame of minimum volatility strategy: \n', df_opt_min_vol_result)

######
###calculate the average year return
df_cum_last = 1
opt_max_sharpe_ratio_return_dailycum = []
for i,k in zip(range(len(rebalance_tradeday_monthly_index)-1),range(len(opt_max_sharpe_weight))):
    
    df_2 = ((df_return.iloc[rebalance_tradeday_monthly_index[i]:rebalance_tradeday_monthly_index[i+1]])+1).cumprod() #one month
    df_cum = df_2*opt_max_sharpe_weight[k]*df_cum_last
    df_cum_sum = df_cum.sum(axis=1)
    opt_max_sharpe_ratio_return_dailycum.append(df_cum_sum)
    df_cum_last= df_cum_sum.iloc[-1]
    
df_last_month_1 = (df_return.iloc[rebalance_tradeday_monthly_index[-1]:] + 1).cumprod()
df_last_month_cum = df_last_month_1*opt_max_sharpe_weight[-1]*df_cum_last
opt_max_sharpe_ratio_return_dailycum.append(df_last_month_cum.sum(axis=1))
df_cum_last = (df_last_month_cum.sum(axis=1)).iloc[-1]

df_opt_max_sharpe_ratio_return_dailycum = pd.concat(opt_max_sharpe_ratio_return_dailycum)
df_opt_max_sharpe_ratio_return_dailycum.index = df_prices.Date.iloc[rebalance_tradeday_monthly_index[0]: ].tolist()

opt_max_sharpe_ratio_return = df_cum_last**(12/len(rebalance_tradeday_monthly_index))-1
print('The return of 4 years for the maximum Sharpe Ratio strategy is: ', '{:.2%}'.format(opt_max_sharpe_ratio_return))

plt.plot(df_opt_max_sharpe_ratio_return_dailycum,label='Max sharpe ratio')
plt.show()
###
df_cum_last_2 = 1
opt_min_vol_return_dailycum = []
for m,n in zip(range(len(rebalance_tradeday_monthly_index)-1),range(len(opt_min_vol_weight))):
    
    df_5 = ((df_return.iloc[rebalance_tradeday_monthly_index[m]:rebalance_tradeday_monthly_index[m+1]])+1).cumprod() #one month
    df_cum = df_5*opt_min_vol_weight[n]*df_cum_last_2
    df_cum_sum = df_cum.sum(axis=1)
    opt_min_vol_return_dailycum.append(df_cum_sum)
    df_cum_last_2= df_cum_sum.iloc[-1]


df_last_month_2 = (df_return.iloc[rebalance_tradeday_monthly_index[-1]:] + 1).cumprod()
df_last_month_cum_2 = df_last_month_2*opt_min_vol_weight[-1]*df_cum_last_2
opt_min_vol_return_dailycum.append(df_last_month_cum_2.sum(axis=1))
df_last_month_2 = (df_last_month_cum_2.sum(axis=1)).iloc[-1]

df_opt_min_vol_return_dailycum = pd.concat(opt_min_vol_return_dailycum)
df_opt_min_vol_return_dailycum.index = df_prices.Date.iloc[rebalance_tradeday_monthly_index[0]: ].tolist()

opt_min_vol_return = df_cum_last_2**(12/len(rebalance_tradeday_monthly_index))-1
print('The return of 4 years for the mimimum volatility strategy is: ', '{:.2%}'.format(opt_min_vol_return))

plt.plot(df_opt_min_vol_return_dailycum, label='Min volatility')
plt.show()

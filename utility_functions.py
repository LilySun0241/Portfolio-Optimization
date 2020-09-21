import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pandas_datareader.data as pdrd
import scipy.optimize as solver
from functools import reduce
from datetime import datetime
from typing import List


#collect data from Yahoo
def collectdata(tickers: List[str], start: datetime, end: datetime, data_source, to_csv: bool = True) -> pd.DataFrame:
    prices = []
    for ticker in tickers:
        adj_close = pdrd.DataReader(name=ticker, start=start, end=end, data_source=data_source)[['Adj Close']]
        adj_close.columns = [ticker]
        prices.append(adj_close)
    df_prices = pd.concat(prices, axis=1)
    df_prices = df_prices.reset_index()

    # output csv
    if to_csv:
        path = './stock_data/' + datetime.now().strftime('%y-%m-%d_%H%M%S') + '.csv'
        df_prices.to_csv(path, index = True, header=True)
    return df_prices

df_prices = collectdata(tickers=['GOOG', 'MSFT', 'AMZN', 'AAPL', 'NFLX'], start=datetime(2015,1,1), end=datetime(2019,12,31), data_source = default_data_source)


#get returns for the tickers
def getreturns(df_prices) -> pd.DataFrame:
    prices_tickers = df_prices.iloc[:, 1:]
    returns_daily = prices_tickers.pct_change()
    returns_year = returns_daily.mean() * 252
    return returns_daily


#maximum Sharpe Ratio Strategy
def max_sharpe_ratio_strat(df_oneyear_return: pd.DataFrame, risk_free_rate = default_risk_free_rate) -> pd.DataFrame:

    df_1_mean_return = df_oneyear_return.mean() #daily mean return in the one year
    df_1_cov = df_oneyear_return.cov()
    assets_num = len(df_oneyear_return.columns)
    bounds = tuple((0.0, 1.0) for i in range(assets_num))
    guess = np.array(assets_num*[1/assets_num])
    
    def negative_sharpe(w, df_1_mean_return, df_1_cov, risk_free_rate):
        vol = np.sqrt(reduce(np.dot, [w.T, df_1_cov, w])) * np.sqrt(252)
        ret = np.dot(w, df_1_mean_return) * 252
        return -(ret-risk_free_rate)/vol
    
    args = (df_1_mean_return, df_1_cov, risk_free_rate)
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
    min_neg_sharpe = solver.minimize(fun=negative_sharpe, x0=guess, args=args, constraints=constraints, bounds=bounds, method='SLSQP')
    return -min_neg_sharpe.fun, min_neg_sharpe.x


#minimum volatility strategy
def min_vol_strat(df_oneyear_return: pd.DataFrame) -> pd.DataFrame:
    df_1_mean_return = df_oneyear_return.mean() #daily mean return in the one year
    df_1_cov = df_oneyear_return.cov()
    assets_num = len(df_oneyear_return.columns)
    bounds = tuple((0.0, 1.0) for i in range(assets_num))
    guess = np.array(assets_num*[1/assets_num])
    
    target_return = np.linspace(max(df_1_mean_return)*252, min(df_1_mean_return)*252, 1000)
    def vol(w):
        return np.sqrt(reduce(np.dot, [w.T, df_1_cov, w])) * np.sqrt(252)
        
    min_vol_result = float('inf')
    min_vol_weight = []
    min_vol_return = []
    for i in target_return:
        constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}, {'type': 'eq', 'fun': lambda x: sum(x*df_1_mean_return)*252 - i}]
        min_vol = solver.minimize(fun=vol, x0=guess, constraints=constraints, bounds=bounds, method='SLSQP')
        if min_vol.fun < min_vol_result:
            min_vol_result = min_vol.fun
            min_vol_weight = min_vol.x
            min_vol_return = i
        else:
            break
        
    return min_vol_return, min_vol_result, min_vol_weight


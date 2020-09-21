import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pandas_datareader.data as pdrd
import scipy.optimize as solver
from functools import reduce
from datetime import datetime
from typing import List

default_data_source = 'yahoo'
default_risk_free_rate = 0.02
to_csv = False

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
        path = './stock_historical_data/' + datetime.now().strftime('%y-%m-%d_%H%M%S') + '.csv'
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
    
    target_return = np.linspace(max(df_1_mean_return)*252, min(df_1_mean_return)*252, 50)
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

#random weight percentage generated
def portfolio_comb(df_prices, returns_daily, total_count: int = 1000, r_rf: float = 0.02) -> pd.DataFrame:
    ticker_std = returns_daily.std(ddof=0)
    ticker_cov = returns_daily.cov()
    asset_num = len(df_prices.columns)-1
    return_comb = []
    std_comb = []
    sharpe_ratio = []
    ticker_weight = []
    for i in range(total_count):
        w = np.random.random(asset_num)
        w /= sum(w)
        return_comb_i = np.dot(returns_daily.mean(), w) * 252
        return_comb.append(return_comb_i)
        std_comb_i = np.sqrt(np.dot(w, np.dot(w, ticker_cov)) * 252)
        std_comb.append(std_comb_i)
        sharpe_ratio_i = (return_comb_i - r_rf)/std_comb_i
        sharpe_ratio.append(sharpe_ratio_i)
        ticker_weight.append(w)
        
    df_portfolio = pd.DataFrame({'returns_combine': return_comb, 'volatility_combine': std_comb, 'Sharpe Ratio': sharpe_ratio})
    for index, asset in enumerate(df_prices.columns[1:]):
        df_portfolio[asset + ' Weight'] = [w[index] for w in ticker_weight]

    return df_portfolio

# Function for evaluating statistic for a portfolio given the list of rebalance day and allocations of assets
def get_portfolio_stat(prices: pd.DataFrame, weight: list, rebalance_tradeday_pos: list, risk_free_rate: float = 0.0) -> tuple:
  # Get the portfolio cumulative return history
  cumulative_ret_history = get_port_cumulative_return(prices, weight, rebalance_tradeday_pos)
  
  # Get the drawdown of portfolio
  drawdown_history = get_drawdown(cumulative_ret_history)
  max_daily_drawdown = min(drawdown_history)
  df = drawdown_history.reset_index(drop=True)
  max_drawdown_duration = df[df==0.0].reset_index().diff()['index'].max()
  
  # Portfolio daily returns under this strategy
  port_ret = cumulative_ret_history.pct_change()[1:]
  
  annualized_port_return = cumulative_ret_history[-1] ** (252/(len(port_ret))) - 1
  annualized_port_std = np.sqrt(252) * port_ret.std() 
  sharpe_ratio = annualized_port_return - risk_free_rate / annualized_port_std
  sortino_ratio = get_sortino_ratio(port_ret, 0.0, risk_free_rate)
  
  port_stat = {'annual_ret': annualized_port_return,
              'annual_std': annualized_port_std,
              'cumulative_ret': cumulative_ret_history[-1] - 1,
              'max_daily_drawdown': max_daily_drawdown,
              'max_drawdown_duration': max_drawdown_duration,
              'sharpe_ratio': sharpe_ratio,
              'sortino_ratio': sortino_ratio}
  # rounding values to 4 decimal places
  for k, v in port_stat.items():
    port_stat[k] = round(v, 4)
  return port_stat, cumulative_ret_history, drawdown_history

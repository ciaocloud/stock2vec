#!/usr/bin/env python
# coding: utf-8

'''
Created on Dec 8, 2018

@author: wangxing
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import datetime as dt
# from DQTrade.util import *


# def get_df(dates, fname='./data/gemini_BTCUSD_1hr.csv', colnames = ['Close', 'Low', 'High']):
# #     dates = pd.date_range(start_time, end_time, freq='H')
#     df_temp = pd.read_csv(fname, index_col='Date', parse_dates=True, usecols=['Date'] + colnames, na_values=['nan'])
#     df = pd.DataFrame(index=dates)
#     df = df.join(df_temp)
#     df.dropna(inplace=True)
# #     df.fillna(method='ffill', inplace=True)
# #     df.fillna(method='bfill', inplace=True)  
#     return df

def compute_momentum(prices, lookback=14):
    '''
    m[t] = p[t] / p[t - window] - 1
    '''
    momentum = (prices / prices.shift(lookback).values) - 1
    return momentum

def simple_moving_average(prices, lookback=14):
    rolling_mean = prices.rolling(window=lookback).mean()
    return rolling_mean
    
# def compute_sma(prices, lookback=14, col='Close', gen_plot=False):
def compute_sma(prices, lookback=14, symbol='BTC', gen_plot=False):
    rolling_mean = simple_moving_average(prices, lookback)
    sma = prices / rolling_mean.values 
#     print(sma)
    if gen_plot:
        norm_prices = prices / prices.ix[0]
        ax = norm_prices.plot(label='price', title = "Price/SMA Ratio")
        norm_roll = rolling_mean / prices.ix[0]
        norm_roll.plot(label='SMA', ax=ax)
        sma.plot(label='Price/SMA', ax=ax)
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Closing Price')
        ax.legend(loc='best')
        plt.show()
#         plt.savefig('sma.png')
    sma_df =sma.to_frame(name=symbol)
    return sma_df

def compute_bbp(prices, lookback=14, symbol='BTC', gen_plot=False):
    roll_mean = prices.rolling(window=lookback, min_periods=lookback).mean()
    roll_std = prices.rolling(window=lookback, min_periods=lookback).std()
    lower_band = roll_mean - 2 * roll_std
    upper_band = roll_mean + 2 * roll_std
    bbp = (prices - lower_band) / (upper_band - lower_band)
    if gen_plot:
        ax = (prices/prices.ix[0]).plot(label='price', title = "Bollinger Bands", lw=1)
        (roll_mean/prices.ix[0]).plot(label='rolling mean', ax=ax)
        (upper_band/prices.ix[0]).plot(ls='--', label='upper band', ax=ax)
        (lower_band/prices.ix[0]).plot(ls='--', label='lower band', ax=ax)
#         bbp.plot(label='%B', ax=ax)
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Closing Price')
        ax.legend(loc='best')
        plt.show()
#         plt.savefig('bbp.png')
    bbp_df = bbp.to_frame(name=symbol)
    return bbp_df

def compute_williams(df, lookback=14, symbol='BTC', gen_plot=False):
    close = df['Close']
    low = df['Low']
    high = df['High']
    roll_low = low.rolling(window=lookback, min_periods=lookback).min()
    roll_high = high.rolling(window=lookback, min_periods=lookback).max()
    percentR = -100 * (close - roll_low) / (roll_high - roll_low)
    
    if gen_plot:
        ax = percentR.plot(label='%R', color='g', title='Williams %R')
        ax.axhline(-20, ls='--')
        ax.axhline(-80, ls='--')
        ax.legend(loc='best')
        plt.show()
    percentR_df = percentR.to_frame(name=symbol)
    return percentR_df

def compute_sop(df, lookback=14, symbol='BTC', gen_plot=False):
#     df = get_df(st, et, colnames = ['Close', 'Low', 'High'])
    close = df['Close']
    low = df['Low']
    high = df['High']
    roll_low = low.rolling(window=lookback, min_periods=lookback).min()
    roll_high = high.rolling(window=lookback, min_periods=lookback).max()
    percentK = 100 * (close - roll_low) / (roll_high - roll_low)
    percentD = percentK.rolling(window=3, min_periods=3).mean()
    if gen_plot:
        ax = percentK.plot(label='%K', color='g', title='Stochastic Oscillator')
        percentD.plot(label='%D', color='r', ax=ax)
        ax.axhline(20, ls='--')
        ax.axhline(80, ls='--')
        ax.legend(loc='best')
        plt.show()
#         plt.savefig('sop.png')
#         ax = (close / close.ix[0]).plot(label = 'close')
#         (roll_low / close.ix[0]).plot(label = 'lowest low', ax=ax)
#         (roll_high / close.ix[0]).plot(label = 'highest high', ax=ax)
#         ax.legend(loc='best')
#         ax.set_ylim(1,3)
# 
#         plt.show()
#         plt.savefig('sop.png')
    percentD_df = percentD.to_frame(name=symbol)
    return percentD_df


def compute_port_vals(df_prices, df_trades, start_val=100000, commission=0, impact=0):
    symbols = list(df_trades.columns.values)
    df_vals = pd.DataFrame(index=df_prices.index, columns=symbols+['CASH'])
    cash = start_val
    stocks = pd.Series()
    for sym in symbols:
        stocks[sym] = 0
    for day in df_prices.index:
        if day in df_trades.index:
            cash, stocks = daily_execute(cash, stocks, df_prices, df_trades, day, commission, impact)
        df_vals.loc[day][symbols] = df_prices.loc[day][symbols] * stocks
        df_vals.loc[day]['CASH'] = cash 
    port_vals = df_vals.sum(axis=1)
    port_vals = port_vals / port_vals.ix[0, :]
    return port_vals

def daily_execute(cash, stocks, df_prices, df_trades, day, commission, impact):
    daily_transactions = df_trades.ix[day]
    for sym in df_trades.columns.values:
        num_share = daily_transactions[sym]
        if num_share != 0:
            stocks[sym] += num_share
            price = df_prices.loc[day, sym]
            if num_share > 0:   # BUY order
                order_val = (1. + impact) * price * num_share
            else:               # SELL order
                order_val = (1. - impact) * price * num_share
            cash -= (order_val + commission)
    return cash, stocks

def compute_port_stats(portvals, daily_risk_free_rate=0., num_samples_per_year=252):
    cum_ret = portvals[-1] / portvals[0] - 1    # Cumulative Returns
    
    daily_rets = portvals / portvals.shift(1) - 1
    daily_rets = daily_rets[1:] 
    
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    
    delta = daily_rets - daily_risk_free_rate
    if std_daily_ret == 0:
        sharp_ratio = 0.
    else:
        sharp_ratio = np.sqrt(num_samples_per_year) * delta.mean() / delta.std()
    
    return cum_ret, avg_daily_ret, std_daily_ret, sharp_ratio   
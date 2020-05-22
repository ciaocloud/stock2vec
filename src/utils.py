import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime as dt
import pickle
import os
import time

import seaborn as sns
sns.set()
from matplotlib import style
# style.use('ggplot')

def cat_cont_split(df, maxcard=55, omit_vars=['Date', 'target_price']):
    """Helper function that returns column names of categorical & continuous features from df."""
    cat_feats, cont_feats = [], []
    for col in df:
        if col in omit_vars: 
            continue
        if (df[col].dtype==int or df[col].dtype==float) and \
                                (df[col].unique().shape[0] > maxcard):
            cont_feats.append(col)
        else:
            cat_feats.append(col)
    return cat_feats, cont_feats

class Categorifier:
    ''' Transform categorical features into category types '''
    def apply_train(self, df, cat_vars):
        self.cat_vars = cat_vars
        self.categories = {}
        for v in self.cat_vars:
            df.loc[:, v] = df.loc[:, v].astype('category').cat.as_ordered()
            self.categories[v] = df[v].cat.categories
            
    def apply_test(self, df_test):
        for v in self.cat_vars:
            df_test.loc[:, v] = pd.Categorical(df[v], categories=self.categories[v], ordered=True)
    
            
def df_train_test_split(df, testlen=126, date_ascending=False):
#     n_test = 126
    cut_day = df['Date'].unique()[-testlen]
    last_day = df['Date'].max()
    print("Cut Date: ", cut_day)

    train_df = df[df['Date'] < cut_day].sort_values(by='Date', ascending=date_ascending).reset_index(drop=True)
#     test_df = df[(df['Date'] >= cut_day) & (df['Date'] <= last_day)].sort_values(by='Date', ascending=date_ascending).reset_index(drop=True)
    test_df = df[(df['Date'] >= cut_day) & (df['Date'] < last_day)].sort_values(by='Date', ascending=date_ascending).reset_index(drop=True)
    return train_df, test_df

def eval_ticker(preds_df, ticker):
    idx = preds_df[preds_df.ticker==ticker].index
#     y_plot_pred = preds_df.iloc[idx]['rf_pred'].values[::-1]
#     y_plot_real = preds_df.iloc[idx]['target_price'].values[::-1]
    pred_cols = preds_df.columns.str.endswith('pred')
    real_col = preds_df.columns.str.startswith('target')

    y_plot_pred = preds_df.loc[idx][pred_cols].values[::-1]
    y_plot_real = preds_df.loc[idx][real_col].values[::-1]

    n = len(y_plot_real)
    fig, ax = plt.subplots(1,1, figsize=(20,8))
    x_plot = np.arange(n)
    plt.plot(x_plot, y_plot_real, color="green")
    plt.scatter(x_plot, y_plot_real, c="green", label="Real")
    plt.plot(x_plot, y_plot_pred, color="red")
    plt.scatter(x_plot, y_plot_pred, c="red", label="Prediction")
    plt.title("Real vs Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price $")
    plt.legend()
    plt.show() 


def MAE(y, y_hat):
    return np.mean(np.abs(y_hat-y))

def MAPE(y, y_hat):
#     return np.mean(np.abs(y_hat-y) / (y+5e-1)) * 100
    return np.mean(np.abs(y_hat-y) / y) * 100
                   
def RMSE(y, y_hat):
    return np.sqrt(np.mean((y_hat - y)**2))

def RMSPE(y, y_hat):
#     return np.sqrt(np.mean(((y - y_hat) / (y+5e-1)) ** 2))
    return np.sqrt(np.mean(((y - y_hat) / y) ** 2)) * 100

def scores(y, y_hat):
    return dict(zip(['MAE', 'MAPE', 'RMSE', 'RMSPE'], 
                       [MAE(y, y_hat), MAPE(y, y_hat), RMSE(y, y_hat), RMSPE(y, y_hat)]))

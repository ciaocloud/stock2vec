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
style.use('ggplot')

import utils

def prepareData(datafile='../data_scaled.csv', testlen=126):
    df_all = pd.read_csv(datafile)
    df_all['Date'] = pd.to_datetime(df_all['Date'])

    cat_vars, cont_vars = utils.cat_cont_split(df_all, omit_vars=['Date'])
#     print(len(cat_vars), 'Categorical Features')
#     print(cat_vars)
#     print(len(cont_vars), 'Continuous Features')
#     print(cont_vars)

    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    ohe = OneHotEncoder()
    scaler = StandardScaler()
    cont_cols = scaler.fit_transform(df_all[cont_vars])
    cont_cols.shape

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    cat_features = []
    for v in cat_vars:
        n = len(df_all[v].unique())
#         print(v, n)
        if n > 1: #and n <= 13:
            cat_features.append(v)
#     print(cat_features)
    ohe = OneHotEncoder()
    cat_cols = ohe.fit_transform(df_all[cat_features])
    cat_cols.shape

    cat_tsfm = Pipeline(steps=[
        ('ohe', OneHotEncoder())
    ])
    cont_tsfm = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
    #     ('scaler', StandardScaler())
    ])
    preproc = ColumnTransformer(transformers=[
        ('cont', cont_tsfm, cont_vars),
        ('cat', cat_tsfm, cat_features)
    ])
    
    t0 = time.time()
    tsfm_np = preproc.fit_transform(df_all)
    print("Elapsed: %.1f sec"%(time.time()-t0))

    columns=cont_vars + ohe.get_feature_names(cat_features).tolist()
    df_tsfm = pd.DataFrame(data=tsfm_np.todense(), columns=columns)
    df_tsfm['Date'] = df_all['Date']
    print("Elapsed: %.1f sec"%(time.time()-t0))
   
    train_df, test_df = utils.df_train_test_split(df_tsfm, testlen)

#     print(train_df.shape, test_df.shape, train_df.Date.min(), train_df.Date.max(), test_df.Date.min(), test_df.Date.max())

    dep_var = "target_price"
    cols = cont_vars + ohe.get_feature_names(cat_features).tolist()
    cols.remove(dep_var)
    Xtrain, ytrain = train_df[cols], train_df[dep_var]
    Xtest, ytest = test_df[cols], test_df[dep_var]
    
    print("Finishing data preprocessing, elapsed %d sec..."%(time.time()-t0))
#     del df_tsfm, df, train_df, test_df
    
    return Xtrain, ytrain, Xtest, ytest, df_all

def main(args):
    method = args.method
    Xtrain, ytrain, Xtest, ytest, df_all = prepareData(testlen=args.testlen)
    
    if method=='rf' or method=='randomforest':
        print("Random Forest Model:")
        from sklearn.ensemble import RandomForestRegressor
        
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
        if args.train:
            tic = time.time()
            rf.fit(Xtrain, ytrain)
            elapsed = time.time() - tic
            print("Training Elapsed: {} sec".format(elapsed))
            if args.savemodel is not None:
                pickle.dump(rf, open(args.savemodel, 'wb'))
        else:
            rf = pickle.load(open(args.saved, 'rb'))
            
        pred_train = rf.predict(Xtrain)
        pred_test = rf.predict(Xtest)
            
    elif method=='xgb' or method=='xgboost':
        print("XGBoost Model:")

        import xgboost as xgb

        Dtrain = xgb.DMatrix(Xtrain, label=ytrain)
        Dtest = xgb.DMatrix(Xtest, label=ytest)
        params = {}
        if not args.gpufalse:
            params['tree_method'] = 'gpu_hist'
        params['eta'] = .1
        watchlist = [(Dtest, 'eval'), (Dtrain, 'train')]
        num_round = 100
        bst = xgb.train(params, Dtrain, num_round, watchlist)
        
        pred_train = bst.predict(Dtrain)
        ytrain = Dtrain.get_label()
        print('\n Training metrics:')
        train_scores = utils.scores(ytrain, pred_train)

        pred_test = bst.predict(Dtest)
        ytest = Dtest.get_label()
        print('\n Test metrics:')
        test_scores = utils.scores(ytest, pred_test)

        
    test_preds_df = df_all[['Date', 'ticker', 'target_price']]
    cut_day = df_all['Date'].unique()[-args.testlen]
    last_day = df_all['Date'].max()
    test_preds_df = test_preds_df[(test_preds_df['Date'] >= cut_day) & (test_preds_df['Date'] <= last_day)].sort_values(by='Date', ascending=False).reset_index(drop=True)
    test_preds_df['%s_pred'%method] = pred_test

    if args.savepred:
        test_preds_df.to_csv('Results/%s_test_preds.csv'%method, index=False)
    if args.ticker:
        utils.eval_ticker(test_preds_df, args.ticker)
        


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark models: random forest and XGBoost")
    parser.add_argument('--method', type=str, default="xgboost")
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--testlen', type=int, default=126)
    parser.add_argument('--gpufalse', action='store_true', default=False)
    parser.add_argument('--saved', type=str, default="models/rf_large.mod")
    parser.add_argument('--savepred', action='store_true', default=False)
    parser.add_argument('--ticker', type=str)
    
    args = parser.parse_args()
    
    main(args)
    
    
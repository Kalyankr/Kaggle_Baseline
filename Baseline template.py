#!/usr/bin/env python
# coding: utf-8

# In[1]:




# Imports And Libraries
import time
import warnings
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import gc
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from functools import partial, wraps
from datetime import datetime as dt
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:




# Imports 
train_df = pd.read_csv("../input/train.csv") #dtype= dtypes, nrows=50000)
test_df = pd.read_csv("../input/test.csv")#dtype = dtypes, nrows=50000)
print("Train Shape : ", train_df.shape)
print("Test Shape : ", test_df.shape)


# In[ ]:


def inspect_summary(df):
    """Returns a inspection dataframe"""
    inspect_dataframe = pd.DataFrame({'DataType': df.dtypes, 'Unique values': df.nunique() ,
                  'Number of missing values': df.isnull().sum() ,
                  'Percentage missing': (df.isnull().sum() / len(df)) * 100,
                                      'Memory Usage (MB)':round(df.memory_usage(index=False) / 1024, 2)
                                     }).sort_values(by='Number of missing values', ascending = False)
    inspect_dataframe['Variance'] = df[inspect_dataframe.index].var()
    inspect_dataframe['Mean'] = df[inspect_dataframe.index].mean()
    inspect_dataframe['Min'] = df[inspect_dataframe.index].min()
    inspect_dataframe['Max'] = df[inspect_dataframe.index].max()
    return inspect_dataframe



ins_train = inspect_summary(train_df)
ins_train.transpose()
ins_test = inspect_summary(test_df)
ins_test.transpose()


# Target Analysis

# In[ ]:




target = train_df['Target']
cnt_srs = train_df['Target'].value_counts()
trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Target Count',
    font=dict(size=18))

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="TargetCount")


# Identifying And Grouping Columns

# In[ ]:


categorical_cols = list(ins_train[ins_train['DataType'] == 'category'].index)
numerical_cols = list(ins_train[ins_train['DataType'] != 'category'].index)
binary_cols = []
for col in (numerical_cols):
    if train_df[col].nunique() == 2:
        binary_cols.append(col)
        numerical_cols.remove(col)

for col in (categorical_cols):
    if train_df[col].nunique() == 2:
        binary_cols.append(col)
        categorical_cols.remove(col)
        
print(">> Categorical Columns: {}: \n {} \n\n".format(len(categorical_cols), categorical_cols))
print(">> Numerical Columns: {}: \n{}\n\n".format(len(numerical_cols), numerical_cols))
print(">> Binary Columns: {}:\n{}".format(len(binary_cols), binary_cols))


# 
# Baseline Model And Feature Importances¶
# 

# In[ ]:


params = {'objective': 'binary', 'boosting_type': 'gbdt', 'learning_rate': 0.02, 'max_depth': 8, 'num_leaves': 67, 'n_estimators': 1000, 'bagging_fraction': 0.4, 'feature_fraction': 0.5, 'bagging_freq': 5, 'bagging_seed': 2018, 'min_child_samples': 80, 'min_child_weight': 100.0, 'min_split_gain': 0.1, 'reg_alpha': 0.005, 'reg_lambda': 0.1, 'subsample_for_bin': 25000, 'min_data_per_group': 100, 'max_cat_to_onehot': 4, 'cat_l2': 25.0, 'cat_smooth': 2.0, 'max_cat_threshold': 32, 'random_state': 1, 'silent': True, 'metric': 'auc'}

TARGET = 'column_name'
TARGET_INDEX = 'Index'
def modeling_cross_validation(params, X, y, folds=2):
    clfs = list()
    oof_preds = np.zeros(X.shape[0])
    # Split data with kfold
    kfolds = StratifiedKFold(n_splits=folds, shuffle=False, random_state=42)
    for n_fold, (trn_idx, val_idx) in enumerate(kfolds.split(X, y)):
        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        # LightGBM Regressor estimator
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=-1, eval_metric='auc', #based on problem solving
            early_stopping_rounds=100
        )

        clfs.append(model)
        oof_preds[val_idx] = model.predict(X_valid, num_iteration=model.best_iteration_)
        
    score = roc_auc_score(y, oof_preds)
    print(score)
    return clfs, score

def get_importances(clfs, feature_names):
    importances = pd.DataFrame()
    for i, model in enumerate(clfs, 1):
        # Feature importance
        imp_df = pd.DataFrame({
                "feature": feature_names, 
                "gain": model.booster_.feature_importance(importance_type='gain'),
                "fold": model.n_features_,
                })
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

    importances['gain_log'] = importances['gain']
    mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
    importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])
    importances.sort_values(by='gain', inplace=True, ascending=False)
    importances.to_csv('feature_importances.csv', index=False)
    return importances

def predict_cross_validation(test, clfs):
    sub_preds = np.zeros(test.shape[0])
    for i, model in enumerate(clfs, 1):    
        test_preds = model.predict_proba(test, num_iteration=model.best_iteration_)
        sub_preds += test_preds[:,1]

    sub_preds = sub_preds / len(clfs)
    ret = pd.Series(sub_preds, index=test.index)
    ret.index.name = test.index.name
    return ret
def predict_test_chunk(features, clfs, dtypes, filename='tmp.csv', chunks=100000):
    
    for i_c, df in enumerate(pd.read_csv('../input/test.csv', 
                                         chunksize=chunks, 
                                         dtype=dtypes, 
                                         iterator=True)):
        
        df.set_index(TARGET_INDEX, inplace=True)
        preds_df = predict_cross_validation(df[features], clfs)
        preds_df = preds_df.to_frame(TARGET)

        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=True)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=True)
        del preds_df
        gc.collect()
train_features = list()
train = pd.read_csv('../input/train.csv', nrows=50000, dtype=dtypes).set_index(TARGET_INDEX)
train_features = [f for f in train.columns if f != TARGET]
clfs, score = modeling_cross_validation(params, train[train_features], train[TARGET], folds=5)


# In[ ]:


feature_importances = get_importances(clfs, train_features)
feature_importances.drop_duplicates(subset=['feature'], inplace=True)
data = [go.Bar(
            x= feature_importances.feature.values,
            y= feature_importances.gain.values, marker = dict(
          color = 'gold'
        ), orientation = 'v'
    )]
layout = go.Layout(
    title='Feature Importances LGB Model')
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


train_features = list(feature_importances.feature[:50])
clfs, score = modeling_cross_validation(params, train[train_features], train[TARGET], folds=5)
filename = 'submission_baseline.csv'
predict_test_chunk(train_features, clfs, dtypes, filename=filename, chunks=100000)






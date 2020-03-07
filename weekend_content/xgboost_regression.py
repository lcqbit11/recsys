#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xgboost as xgb
import numpy as np
import pandas as pd
import tensorflow as tf

data_dir = '/Users/changqingli/pycharm_projects/algorithms/weekend_content/'
train_data_file = 'data/happiness_train_abbr.csv'
test_data_dile = 'data/happiness_test_abbr.csv'

df_train = pd.read_csv(data_dir+train_data_file)
df_test = pd.read_csv(data_dir+test_data_dile)

df_train = df_train.drop('survey_time', 1).drop('id', 1)
df_train = df_train[df_train['happiness'] != -8]
df_train = df_train.fillna(0)

df_test = df_test.drop('survey_time', 1)
df_test = df_test.fillna(0)

df_train_train = df_train.sample(frac=0.8, random_state=0, axis=0)
df_train_validation = df_train[~df_train.index.isin(df_train_train.index)]

tr_data = df_train_train.as_matrix()
va_data = df_train_validation.as_matrix()
te_data = df_test.as_matrix()

tr_x_data = tr_data[:, 1:]
tr_y_data = tr_data[:, 0]

va_x_data = va_data[:, 1:]
va_y_data = va_data[:, 0]

mod = xgb.XGBRegressor(max_depth=6,
                       learning_rate=0.15,
                       n_estimators=100,
                       objective='reg:squarederror',
                       booster='gbtree',
                       gamma=5)


mod.fit(tr_x_data, tr_y_data)

xgb.DMatrix()



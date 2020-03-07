#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

dir_name = '/Users/changqingli/pycharm_projects/algorithms/weekend_content/o2o/'
offline_file = 'data/ccf_offline_stage1_train.csv'
online_file = 'data/ccf_online_stage1_train.csv'
test_file = 'data/ccf_offline_stage1_test_revised.csv'
sample_file = 'data/sample_submission.csv'

df_off = pd.read_csv(dir_name+offline_file)
df_on = pd.read_csv(dir_name+online_file)
df_test = pd.read_csv(dir_name+test_file)
df_sample = pd.read_csv(dir_name+sample_file, header=None)

df_off['User_id'].unique()
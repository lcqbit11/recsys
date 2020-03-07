#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0.将 预测结果保持小数形式，而不是转换成整数形式  结果从0.5以上优化到0.5以下 0.47381812389478345
1.将 survey_time数据提取出月份、日期、小时三个特征 0.47143198583365375
2.将 family_income，income离散成5段的离散值 0.4730428161760329
3.将 birth 转换成年龄值age 0.47143198583365375
4.将 family_income，income离散成5段的离散值，并且birth 转换成年龄值age，都加上，0.47241532744595904
6.将 age 离散化成7段年龄段  0.47619062169156706
7.将 survey_time数据提取出一周的第几天、是否周末 两个特征 0.47097871771026456
8.将 family_income、income中小于0的数字转换为0 0.4708401593321734
9.使用全量特征训练 0.466104881012437
10.将 family_income，income提取出来家庭收入占个人收入的比例特征，0.46458776964941884
11.调参：
    1.将max_depth由6调整为8，0.46458776964941884-》0.4623970704927228
    2.将n_estimators由100调整为200  0.4623970704927228-》0.462396775672775
    3.将min_child_weight由1-》2-》3  0.462396775672775-》0.4616176256431441-》0.46051453956155897
    4.将subsample由1调整为0.6  0.46051453956155897-》0.4552454001761164
    5.将colsample_bytree由1调整为0.6  0.4552454001761164->0.4520618951047676
12.将所有的训练样本全部用作训练集，不再分出一部分作为验证机集，以期在测试集上能取得更好的效果
13.将 survey_time数据提取出季度、年份特征 0.4520618951047676-》0.45104197365888876
"""

import xgboost as xgb
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import KFold

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

data_dir = '/Users/changqingli/pycharm_projects/algorithms/weekend_content/'
train_data_file = 'data/happiness_train_complete.csv'
test_data_file = 'data/happiness_test_complete.csv'

df_train = pd.read_csv(data_dir + train_data_file, encoding='gb2312')
df_test = pd.read_csv(data_dir + test_data_file, encoding='gb2312')

df_train = df_train.drop('id', 1)
df_train = df_train[df_train['happiness'] != -8]
df_train = df_train[pd.to_datetime(df_train.survey_time).dt.month != 2]

# survey_time数据提取出月份、日期、小时三个特征
df_train['year'] = pd.to_datetime(df_train.survey_time).dt.year
df_train['quarter'] = pd.to_datetime(df_train.survey_time).dt.quarter
df_train['month'] = pd.to_datetime(df_train.survey_time).dt.month
df_train['day'] = pd.to_datetime(df_train.survey_time).dt.day
df_train['hour'] = pd.to_datetime(df_train.survey_time).dt.hour
df_train['dayofweek'] = pd.to_datetime(df_train.survey_time).dt.dayofweek + 1
df_train['is_weekend'] = (pd.to_datetime(df_train.survey_time).dt.dayofweek + 1).apply(
    lambda x: 1 if x in (6, 7) else 0)
df_test['year'] = pd.to_datetime(df_test.survey_time).dt.year
df_test['quarter'] = pd.to_datetime(df_test.survey_time).dt.quarter
df_test['month'] = pd.to_datetime(df_test.survey_time).dt.month
df_test['day'] = pd.to_datetime(df_test.survey_time).dt.day
df_test['hour'] = pd.to_datetime(df_test.survey_time).dt.hour
df_test['dayofweek'] = pd.to_datetime(df_test.survey_time).dt.dayofweek + 1
df_test['is_weekend'] = (pd.to_datetime(df_test.survey_time).dt.dayofweek + 1).apply(lambda x: 1 if x in (6, 7) else 0)
# 去除 survey_time数据
df_train = df_train.drop('survey_time', 1)
df_test = df_test.drop('survey_time', 1)
# 是否是共产党员
# df_train['join_party'] =
# 家庭收入是个人收入的比例
df_train['income_rate'] = df_train.apply(
    lambda row: row['family_income'] / row['income'] if row['family_income'] * row['income'] > 0 else 0, 1)
df_test['income_rate'] = df_test.apply(
    lambda row: row['family_income'] / row['income'] if row['family_income'] * row['income'] > 0 else 0, 1)
# 将family_income、income小于0的值转换为0值
df_train['family_income'] = df_train['family_income'].apply(lambda x: x if x > 0 else 0)
df_train['income'] = df_train['income'].apply(lambda x: x if x > 0 else 0)
df_test['family_income'] = df_test['family_income'].apply(lambda x: x if x > 0 else 0)
df_test['income'] = df_test['income'].apply(lambda x: x if x > 0 else 0)
# family_income，income离散成5段的离散值
family_income_quantile = df_train.family_income.quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).values
df_train['family_income_stage'] = pd.cut(df_train.family_income, family_income_quantile, labels=[0, 1, 2, 3, 4])
income_quantile = df_train.income.quantile([0, 0.33, 0.66, 1]).values
df_train['income_stage'] = pd.cut(df_train.income, income_quantile, labels=[0, 1, 2])
family_income_quantile = df_test.family_income.quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).values
df_test['family_income_stage'] = pd.cut(df_test.family_income, family_income_quantile, labels=[0, 1, 2, 3, 4])
income_quantile = df_test.income.quantile([0, 0.33, 0.66, 1]).values
df_test['income_stage'] = pd.cut(df_test.income, income_quantile, labels=[0, 1, 2])
# birth 转换成年龄值
df_train['age'] = df_train['year'] - df_train['birth']
df_train = df_train.drop('birth', 1)
# df_train['age'] = pd.cut(df_train.age, [0, 20, 30, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4, 5, 6])
df_test['age'] = df_test['year'] - df_test['birth']
df_test = df_test.drop('birth', 1)


# df_test['age'] = pd.cut(df_test.age, [0, 20, 30, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4, 5, 6])

# 收入分组
def income_cut(x):
    if x < 0:
        return 0
    elif 0 <= x < 1200:
        return 1
    elif 1200 < x <= 10000:
        return 2
    elif 10000 < x < 24000:
        return 3
    elif 24000 < x < 40000:
        return 4
    elif 40000 <= x:
        return 5


df_train = df_train.drop('property_other', 1).drop('edu_other', 1).drop('invest_other', 1)
df_test = df_test.drop('property_other', 1).drop('edu_other', 1).drop('invest_other', 1)

df_train = df_train.fillna(0)
df_test = df_test.fillna(0)
df_train_train = df_train
df_train_train = df_train.sample(frac=0.8, random_state=0, axis=0)
df_train_validation = df_train[~df_train.index.isin(df_train_train.index)]

tr_data = df_train_train.as_matrix()
va_data = df_train_validation.as_matrix()
te_data = df_test.as_matrix()

tr_x_data = tr_data[:, 1:]
tr_y_data = tr_data[:, 0]

va_x_data = va_data[:, 1:]
va_y_data = va_data[:, 0]

mod = xgb.XGBRegressor(max_depth=8,
                       learning_rate=0.1,
                       n_estimators=200,
                       objective='reg:linear',
                       booster='gbtree',
                       gamma=5,
                       reg_lambda=10,
                       min_child_weight=3,
                       subsample=0.6,
                       colsample_bytree=0.6)

mod.fit(tr_x_data, tr_y_data)

va_pred_data = mod.predict(va_x_data)
np.mean((va_pred_data - va_y_data) * (va_pred_data - va_y_data))

pred_data = te_data[:, 1:]
pred_res = mod.predict(pred_data)

import csv

result = 'submission_908_xgb_tiaocan_7.csv'
with open(data_dir + result, 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['id', 'happiness'])
    for i in range(len(te_data)):
        writer.writerow([int(te_data[i][0]), pred_res[i]])

from xgboost import plot_importance
import matplotlib.pyplot as plt

plot_importance(mod)
plt.show()

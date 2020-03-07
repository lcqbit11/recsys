#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
1.输入特征格式为libsvm格式的，特征数量50+
2.样本label取值为0，1，二分类问题，样本数量30w+
"""

import tensorflow as tf
import pandas as pd

class DeepFM(object):
    def __init__(self, train_data, eval_data, test_data, feat_name):
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.feat_name = feat_name

    def load_data(self):
        df_train = pd.read_csv(config.TRAIN_FILE)
        df_test = pd.read_csv(config.TEST_FILE)
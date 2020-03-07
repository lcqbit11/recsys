#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    meta_data = load_breast_cancer()

    data = meta_data.data
    label = meta_data.target
    train_X, test_X, train_y, test_y = train_test_split(data, label, test_size=0.2)

    lr = LogisticRegression()

    # 直接训练方式
    lr.fit(train_X, train_y)
    score = lr.score(test_X, test_y)
    print(score)

    # 交叉验证训练方式
    score1 = cross_val_score(lr, data, label, cv=5, scoring='accuracy')
    print(score1)
    print(score1.mean())
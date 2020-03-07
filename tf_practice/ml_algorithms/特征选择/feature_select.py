#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.feature_selection import RFE
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


if __name__ == "__main__":
    raw_data = load_breast_cancer()
    data = raw_data.data
    label = raw_data.target
    train_X, test_X, train_y, test_y = train_test_split(data, label, test_size=0.3)

    clf = SVR(kernel='linear')
    selector = RFE(clf, n_features_to_select=10, step=1)
    selector.fit(train_X, train_y)

    score = selector.score(test_X, test_y)

    print(score)
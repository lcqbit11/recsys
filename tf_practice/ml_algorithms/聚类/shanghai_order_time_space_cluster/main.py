#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
:parameter
--------------------
data: n-array like
[
[1.0 2.3 2.1 0.2]
[0.0 2.4 1.1 0.7]
]
每个元素都是有表示距离的属性的，对于没有距离属性的要抓化成具有距离属性的
"""

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir_name = '/Users/changqingli/data/time_space_cluster/'
feat_file_mini = 'fs_mini'
feat_file_all = 'fs_all'

f_data = []
user_info = []

with open(dir_name + feat_file_mini, 'r') as f:
    for line in f:
        line = line.strip().split("\t")[:-1]
        line.insert(1, line[0][:4])  # _year
        line.insert(2, line[0][5:7])  # _month
        line.insert(3, line[0][8:10])  # _date
        line.insert(4, line[0][11:13])  # _hour
        f_data.append(line[1:-1])  # [_year,_month,_date,_hour,latitude,longitude]
        user_info.append([line[0], line[-2], line[-1]])
scale = MinMaxScaler()
scale.fit(f_data)
f_data_transform = scale.transform(f_data)
f_data_transform_train, f_data_transform_test = train_test_split(f_data_transform, test_size=0.2, random_state=0)


def key_count(cluster_y):
    m = {}
    for i in cluster_y:
        if i not in m:
            m.setdefault(i, 0)
        m[i] += 1
    res = sorted(m.items(), key=lambda x: x[1], reverse=True)
    return res


k_means = KMeans()
k_means.fit(f_data_transform_train)
cluster_train_counts = key_count(k_means.labels_)
print(cluster_train_counts)

test_label = k_means.predict(f_data_transform_test)
cluster_test_counts = key_count(test_label)
print(cluster_test_counts)
print('f_data_transform_test,', f_data_transform_test[0])
print('test_label,', test_label)
cluster_test = np.hstack((f_data_transform_test, test_label))
print('cluster_test[0],', cluster_test[0])

tsne = TSNE()
tsne.fit_transform(cluster_test)  # 进行数据降维,降成两维
tsne = pd.DataFrame(tsne.embedding_, index=cluster_test.index) # 转换数据格式

d=tsne[cluster_test[-1] == 0]
plt.plot(d[0],d[1], 'r.')
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 200)

"""
'gender', 'agerange', 'occupation', 'marital', 'parentkid','salarylevel', 'saunterlevel', 'edulevel', 'mobiletype'
"""


if __name__ == "__main__":
    TRAIN = True
    # TEST = False

    # 数据格式：['uid', 'gender', 'agerange', 'occupation', 'marital', 'parentkid','salarylevel', 'saunterlevel', 'edulevel', 'mobiletype']
    work_dir = '/Users/changqingli/pycharm_projects/embedding_proj/1120/'
    input_file = "/Users/changqingli/pycharm_projects/embedding_proj/1120/order_user_profile_data_30w"
    df = pd.read_table(input_file, header=None)
    columns = ['uid', 'gender', 'agerange', 'occupation', 'marital', 'parentkid','salarylevel', 'saunterlevel', 'edulevel', 'mobiletype']
    df.columns = columns

    format_data = df[['gender', 'agerange', 'occupation', 'marital', 'parentkid', 'salarylevel', 'saunterlevel', 'edulevel', 'mobiletype']].values
    scale = MinMaxScaler()
    transform_data = scale.fit_transform(format_data)
    # train_X, test_X = train_test_split(format_data, test_size=0.2)


    def color_set(data_label, N):
        color_unique = []
        color = []
        for i in range(N):
            color_unique.append(i*i)
        for i in range(len(data_label)):
            color.append(color_unique[data_label[i]])

        return color

    if TRAIN:
        print(len(transform_data))
        print('train cluster model_v3...')
        # DBSCAN
        dbs = DBSCAN().fit(transform_data)
        cluster_label = dbs.labels_
        print(set(cluster_label))
        print(len(set(cluster_label)) - 1)
        n_clusters = len(set(cluster_label)) - 1

        # # K-means聚类
        km = KMeans(n_clusters=n_clusters, init='k-means++')
        km.fit(transform_data)
        cluster_label = km.labels_
        cluster_centers = km.cluster_centers_

        if not os.path.exists('./model_v3/'):
            os.mkdir('./model_v3/')
        with open('./model_v3/km_model.pickle', 'wb') as f:
            pickle.dump(km, f)
    else:
        print('dump cluster model_v3...')
        pickle_in = open('./model_v3/km_model.pickle', 'rb')
        km = pickle.load(pickle_in)
        cluster_label = km.predict(transform_data)

        label = [1] * len(df)
        df['cluster_label'] = cluster_label

        max_num_label = list(df['cluster_label'].value_counts().index)[0]
        write_data = df[df['cluster_label'] == max_num_label]

        write_data.to_excel('./user_profile_data_v3.xlsx')

        write_data_columns = list(write_data.columns)
        for i in range(1, len(write_data_columns) - 1):
            print(write_data[write_data_columns[i]].value_counts())

    # cluster_label = np.array([1] * len(df)).reshape()
    # df['cluster_label'] = cluster_label
    # print(df.head())
    # min_dis = float("inf")
    # min_index = 0
    # for i in range(len(cluster_centers)):
    #     tmp = sum(pow(train_X[0] - cluster_cPartition Equal Subset Sumenters[i], 2))
    #     if tmp < min_dis:
    #         min_dis = tmp
    #         min_index = i

    # PCA降维
    # U = PCA(n_components=2).fit_transform(train_X)
    # color = color_set(cluster_label, 32)
    # print('color:', color)
    # plt.scatter(U[:, 0], U[:, 1], marker='o', c=color)
    # plt.show()
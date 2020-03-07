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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 200)


if __name__ == "__main__":
    # 数据格式：['uid', 'gender', 'agerange', 'occupation', 'marital', 'parentkid','salarylevel', 'saunterlevel', 'edulevel', 'mobiletype']
    input_file = "/Users/changqingli/pycharm_projects/embedding_proj/1120/youtube_data1"
    df = pd.read_table(input_file, header=None)
    columns = ['uid', 'gender', 'agerange', 'occupation', 'marital', 'parentkid','salarylevel', 'saunterlevel', 'edulevel', 'mobiletype']
    df.columns = columns

    format_data = df[['gender', 'agerange', 'occupation', 'marital', 'parentkid','salarylevel', 'saunterlevel', 'edulevel', 'mobiletype']].values
    # format_data = df[['gender', 'agerange', 'occupation', 'marital', 'parentkid']].values
    # scale = StandardScaler()
    scale = MinMaxScaler()
    transform_data = scale.fit_transform(format_data)
    train_X, test_X = train_test_split(transform_data, test_size=0.2)

    def color_set(data_label):
        color = []
        for i in range(len(data_label)):
            if data_label[i] == 0:
                color.append('k')
            elif data_label[i] == 1:
                color.append('g')
            elif data_label[i] == 2:
                color.append('r')
            else:
                color.append('b')
        return color

    # K-means聚类
    # km = KMeans(n_clusters=3, init='random')
    # km.fit(train_X)
    # cluster_label = km.labels_
    # cluster_centers = km.cluster_centers_
    # min_dis = float("inf")
    # min_index = 0
    # for i in range(len(cluster_centers)):
    #     tmp = sum(pow(train_X[0] - cluster_cPartition Equal Subset Sumenters[i], 2))
    #     if tmp < min_dis:
    #         min_dis = tmp
    #         min_index = i

    # mean-shift
    ms = MeanShift().fit(train_X)
    cluster_label = ms.labels_
    cluster_centers = ms.cluster_centers_
    print(set(cluster_label))

    # DBSCAN
    # dbs = DBSCAN().fit(train_X)
    # cluster_label = dbs.labels_
    # print(set(cluster_label))


    # t-sne降维
    train_X_embed = TSNE(n_components=2).fit_transform(train_X)
    color = color_set(cluster_label)
    plt.scatter(train_X_embed[:, 0], train_X_embed[:, 1], c=color)
    plt.show()

    # PCA降维
    # U = PCA(n_components=2).fit_transform(train_X)
    # color = color_set(cluster_label)
    # plt.scatter(U[:, 0], U[:, 1], marker='o', c=color)
    # plt.show()

    # LDA 有监督
    # lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(train_X)
    # color = color_set()
    # plt.scatter(lda[:, 0], lda[:, 1], marker='o', c='b')
    # plt.show()
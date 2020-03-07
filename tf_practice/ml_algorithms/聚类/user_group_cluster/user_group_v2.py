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
    TRAIN = False
    # TEST = False

    # 数据格式：['uid', 'gender', 'agerange', 'occupation', 'marital', 'parentkid','salarylevel', 'saunterlevel', 'edulevel', 'mobiletype']
    work_dir = '/Users/changqingli/pycharm_projects/embedding_proj/1120/'
    input_file = "/Users/changqingli/pycharm_projects/embedding_proj/1120/youtube_data1"
    df = pd.read_table(input_file, header=None)
    columns = ['uid', 'gender', 'agerange', 'occupation', 'marital', 'parentkid','salarylevel', 'saunterlevel', 'edulevel', 'mobiletype']
    df.columns = columns
    print('agerange')
    print(df['agerange'].value_counts())
    print('df:', len(df))

    # print("value_counts")
    # print(df['gender'].value_counts())

    format_data = df[['gender', 'agerange', 'occupation', 'marital', 'parentkid','salarylevel', 'saunterlevel', 'edulevel', 'mobiletype']].values
    scale = MinMaxScaler()
    transform_data = scale.fit_transform(format_data)
    print('transform_data:', len(transform_data))
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
        print('train cluster model_v2...')
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

        if not os.path.exists('./model_v2/'):
            os.mkdir('./model_v2/')
        with open('./model_v2/km_model.pickle', 'wb') as f:
            pickle.dump(km, f)
    else:
        print('dump cluster model_v2...')
        pickle_in = open('./model_v2/km_model.pickle', 'rb')
        km = pickle.load(pickle_in)
        cluster_label = km.predict(transform_data)
        print('cluster_label', len(cluster_label))

        label = [1] * len(df)
        df['cluster_label'] = cluster_label

        for col in list(df.columns)[1:-1]:
            print(df[col].value_counts())

        cluster_label_sort = list(df['cluster_label'].value_counts().index)

        count_users_cluster = 0
        count_users_sum = len(df)
        print('总聚类簇数量：', len(cluster_label_sort))
        print('总聚类样本数量：', count_users_sum)
        for t in range(len(cluster_label_sort)):

            cluster_label_value = cluster_label_sort[t]
            write_data = df[df['cluster_label'] == cluster_label_value]
            count_users_cluster += len(write_data)
            count_users_rate = round(float(len(write_data))/count_users_sum, 3)
            # print("当前聚类中心：", cluster_label_value)

            write_data.to_excel('./user_profile_data_v2.xlsx')

            user_profile_value_map = {'gender': {1: '女', 2: '男'},
                                'agerange': {1: '20以下', 2: '20-25', 3: '25-30', 4: '30-35', 5: '35-40', 6: '40以上'},
                                'occupation': {1: '学生', 2: '白领', 3: '未知'},
                                'marital': {1: '未婚', 2: '已婚'},
                                'parentkid': {1: '非亲子', 2: '是亲子'},
                                'salarylevel': {1: '低', 2: '中', 3: '高'},
                                'saunterlevel': {1: '低', 2: '中', 3: '高'},
                                'edulevel': {1: '本科以下', 2: '本科及以上'},
                                'mobiletype': {2: 'ios', 3: 'Android'}
                                }

            user_profile_name_map = {'gender': '性别', 'agerange': '年龄段', 'occupation': '职业',
                                     'marital': '是否已婚', 'parentkid': '是否亲子', 'salarylevel': '收入水平',
                                     'saunterlevel': '活跃频率', 'edulevel': '教育水平', 'mobiletype': '手机类型'}

            write_data_columns = list(write_data.columns)
            user_module = '聚类中心' + str(cluster_label_value) + '(' + str(count_users_rate) + '):'
            for i in range(1, len(write_data_columns) - 1):
                # print(write_data_columns[i])
                attr_name = write_data_columns[i]

                attr_label = write_data[write_data_columns[i]].value_counts().index.tolist()
                attr_num = write_data[write_data_columns[i]].value_counts().tolist()
                # print(user_profile_map[attr_name][attr_label[0]] + '\t' + str(attr_num[0]))
                rate = round(attr_num[0]/sum(attr_num), 3)
                user_module += user_profile_name_map[write_data_columns[i]] + '=' + user_profile_value_map[attr_name][attr_label[0]] + '(' + str(rate) + ')' + ','
                # print(user_profile_name_map[write_data_columns[i]] + '=' + user_profile_value_map[attr_name][attr_label[0]])
                # for j in range(len(attr_label)):
                #     print(user_profile_map[attr_name][attr_label[j]] + '\t' + str(attr_num[j]))
            print(user_module[:-1])
        print(float(count_users_cluster)/count_users_sum)



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
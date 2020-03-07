#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 200)


if __name__ == "__main__":
    file_name = '/Users/changqingli/data/scene_data_7d.xlsx'
    write_dir = './scene_file_7d/'
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    data = pd.read_excel(file_name)
    columns = ['gender', 'age_level', 'is_parent_child', 'is_married', 'occupation', 'num']
    data = data[columns]
    data = data.loc[(data['is_parent_child'] != '非亲子') & (data['occupation'] != '未知')]
    data = data.fillna('无标签')

    scene_count = 0

    # 使用5个人群因子
    print("使用5个人群因子")
    data = data.sort_values('num', ascending=False)
    data_5 = data.copy()
    columns_5 = columns.copy()

    len_5 = len(data_5)
    sum_5 = data_5['num'].sum()
    columns_5.append('sum_num')
    data_5.reindex(columns=columns_5)
    data_5['sum_num'] = [sum_5 for _ in range(len_5)]
    data_5.eval("rate = num / sum_num", inplace=True)
    data_5_values = data_5.values
    # print(data_5_values[:, :-3])
    print(data_5)
    null_count = 0
    print('len data_5:', len(data_5))
    data_5.to_excel(write_dir + 'file_5.xlsx')
    scene_count += len(data_5)

    # 使用4个人群因子
    print("使用4个人群因子")
    count = 0
    for i in range(len(columns)-1):
        data_4 = data.copy()
        columns_4 = columns.copy()
        columns_4.pop(i)
        data_4 = data_4[columns_4]

        columns_4.pop()  # 去除num
        data_4_ = data_4.groupby(columns_4)['num'].sum().reset_index()
        data_4_ = data_4_.sort_values('num', ascending=False)
        sum_4 = data_4_['num'].sum()
        # print(sum_4)
        data_4_['sum_num'] = [sum_4 for _ in range(len(data_4_))]
        data_4_.eval("rate = num / sum_num", inplace=True)
        data_4_.to_excel(write_dir + 'file_4_%d.xlsx' % (count))
        count += 1
        scene_count += len(data_4_)

    # 使用3个人群因子
    print("使用3个人群因子")
    count = 0
    for i in range(len(columns)-1):
        for j in range(i+1, len(columns)-1):
            data_3 = data.copy()
            columns_3 = columns.copy()
            columns_3.pop(i)
            columns_3.pop(j-1)
            data_3 = data_3[columns_3]
            # print(columns_3)

            columns_3.pop()  # 去除num
            data_3_ = data_3.groupby(columns_3)['num'].sum().reset_index()
            data_3_ = data_3_.sort_values('num', ascending=False)
            sum_3 = data_3_['num'].sum()
            # print(sum_3)
            data_3_['sum_num'] = [sum_3 for _ in range(len(data_3_))]
            data_3_.eval("rate = num / sum_num", inplace=True)
            data_3_.to_excel(write_dir + 'file_3_%d.xlsx' % (count))
            count += 1
            scene_count += len(data_3_)

    # 使用2个人群因子
    print("使用2个人群因子")
    count = 0
    for i in range(len(columns)-1):
        for j in range(i+1, len(columns)-1):
            data_2 = data.copy()
            columns_2 = columns.copy()
            data_2 = data_2[[columns_2[i], columns_2[j], columns_2[-1]]]

            # columns_2.pop()  # 去除num
            data_2_ = data_2.groupby([columns_2[i], columns_2[j]])['num'].sum().reset_index()
            data_2_ = data_2_.sort_values('num', ascending=False)
            sum_2 = data_2_['num'].sum()
            # print(sum_2)
            data_2_['sum_num'] = [sum_2 for _ in range(len(data_2_))]
            data_2_.eval("rate = num / sum_num", inplace=True)
            data_2_.to_excel(write_dir + 'file_2_%d.xlsx' % (count))
            count += 1
            scene_count += len(data_2_)

    # 使用1个人群因子
    print("使用1个人群因子")
    count = 0
    for i in range(len(columns) - 1):
        data_1 = data.copy()
        columns_1 = columns.copy()
        data_1 = data_1[[columns_1[i], columns_1[-1]]]
        # print([columns_1[i]])

        # columns_4.pop()  # 去除num
        data_1_ = data_1.groupby(columns_1[i])['num'].sum().reset_index()
        data_1_ = data_1_.sort_values('num', ascending=False)
        sum_1 = data_1_['num'].sum()
        # print(sum_1)
        data_1_['sum_num'] = [sum_1 for _ in range(len(data_1_))]
        data_1_.eval("rate = num / sum_num", inplace=True)
        data_1_.to_excel(write_dir + 'file_1_%d.xlsx' % (count))
        count += 1
        scene_count += len(data_1_)

    print('scene_count:', scene_count)
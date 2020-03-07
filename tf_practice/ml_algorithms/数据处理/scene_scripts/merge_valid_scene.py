#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 200)
"""
对5个人群因子、4个人群因子、3个人群因子、2个人群因子、1个人群因子的场景进行merge
"""


if __name__ == "__main__":
    scene_set = set()
    scene_map = dict()
    write_dir = './scene_file_7d/'

    # 5因子场景
    data_5 = pd.read_excel(write_dir + 'file_5.xlsx')
    data_5_values = data_5.values
    scene_name_5 = dict()
    for i in range(len(data_5_values)):
        tmp = ''
        if data_5_values[i].tolist()[1:-3].count('无标签') < 5:
            for j in range(1, len(data_5_values[i].tolist())-3):
                if data_5_values[i][j] != '无标签':
                    tmp += str(data_5_values[i][j]) + '|'
            scene_name_5[tmp[:-1]] = round(data_5_values[i][-1], 4)
    scene_map['5'] = scene_name_5
    scene_set.update(scene_name_5.keys())
    print(len(scene_set))

    # 4因子场景

    for file_index in range(5):
        data_4 = pd.read_excel(write_dir + 'file_4_' + str(file_index) + '.xlsx')
        data_4_values = data_4.values
        scene_name_4 = dict()
        for i in range(len(data_4_values)):
            tmp = ''
            if data_4_values[i].tolist()[1:-3].count('无标签') < 5:
                for j in range(1, len(data_4_values[i].tolist())-3):
                    if data_4_values[i][j] != '无标签':
                        tmp += str(data_4_values[i][j]) + '|'
                scene_name_4[tmp[:-1]] = round(data_4_values[i][-1], 4)
        scene_set.update(set(scene_name_4.keys()))
        scene_map['4_' + str(file_index)] = scene_name_4
    print(len(scene_set))

    # 3因子场景
    for file_index in range(10):
        data_3 = pd.read_excel(write_dir + 'file_3_' + str(file_index) + '.xlsx')
        data_3_values = data_3.values
        scene_name_3 = dict()
        for i in range(len(data_3_values)):
            tmp = ''
            if data_3_values[i].tolist()[1:-3].count('无标签') < 5:
                for j in range(1, len(data_3_values[i].tolist())-3):
                    if data_3_values[i][j] != '无标签':
                        tmp += str(data_3_values[i][j]) + '|'
                scene_name_3[tmp[:-1]] = round(data_3_values[i][-1], 4)
        scene_map['3_' + str(file_index)] = scene_name_3
        scene_set.update(set(scene_name_3.keys()))
    print(len(scene_set))

    # 2因子场景
    for file_index in range(10):
        data_2 = pd.read_excel(write_dir + 'file_2_' + str(file_index) + '.xlsx')
        data_2_values = data_2.values
        scene_name_2 = dict()
        for i in range(len(data_2_values)):
            tmp = ''
            if data_2_values[i].tolist()[1:-3].count('无标签') < 5:
                for j in range(1, len(data_2_values[i].tolist())-3):
                    if data_2_values[i][j] != '无标签':
                        tmp += str(data_2_values[i][j]) + '|'
                scene_name_2[tmp[:-1]] = round(data_2_values[i][-1], 4)
        scene_map['2_' + str(file_index)] = scene_name_2
        scene_set.update(set(scene_name_2.keys()))
    print(len(scene_set))

    # 1因子场景
    for file_index in range(5):
        data_1 = pd.read_excel(write_dir + 'file_1_' + str(file_index) + '.xlsx')
        data_1_values = data_1.values
        scene_name_1 = dict()
        for i in range(len(data_1_values)):
            tmp = ''
            if data_1_values[i].tolist()[1:-3].count('无标签') < 5:
                for j in range(1, len(data_1_values[i].tolist())-3):
                    if data_1_values[i][j] != '无标签':
                        tmp += str(data_1_values[i][j]) + '|'
                scene_name_1[tmp[:-1]] = round(data_1_values[i][-1], 4)
        scene_map['1_' + str(file_index)] = scene_name_1
        scene_set.update(set(scene_name_1.keys()))

    print(len(scene_set))

    scene_list = list(scene_set)
    scene_list.sort()
    print("scene list--------")
    valid_index = 0
    user_profile_map = {'男': '性别', '女': '性别',
                        '20以下': '年龄', '20-25': '年龄', '25-30': '年龄', '30-35': '年龄', '35-40': '年龄', '40以上': '年龄',
                        '是亲子': '是否亲子', '非亲子': '是否亲子',
                        '已婚': '是否已婚', '未婚': '是否已婚',
                        '学生': '职业', '白领': '职业'}
    for scene in scene_list:
        if not scene or scene == '男|30-35|已婚|学生' or scene == '男|30-35|未婚|学生':
            continue
        scene_ = scene.split('|')
        if (scene_.count('是亲子') and scene_.count('学生')) or \
             (scene_.count('40以上') and scene_.count('学生')):
            continue
        for i in range(len(scene_)):
            scene_[i] = user_profile_map[scene_[i]] + ':' + scene_[i]
        scene = ','.join(scene_)
        print('人群编号' + str() + str(valid_index) + '\t' + scene)
        valid_index += 1
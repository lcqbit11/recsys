#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为深度网络模型生成训练数据，对于生成的csv格式的数据，
需要去除column行即第一行为列名称的行，这样训练过程才不会报错，否则会报错。
"""
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 200)


if __name__ == '__main__':
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
        'income_bracket'
    ]  # 14 feature + 1 label
    _CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                            [0], [0], [0], [''], ['']]
    train_data = pd.DataFrame([[1, 'workclass_a|workclass_b', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '>50K'],
                               [3, 'workclass_a|workclass_b', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '>50K'],
                               [2, 'workclass_a|workclass_b', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '>50K'],
                               [2, 'workclass_a|workclass_b', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '>50K'],
                               [5, 'workclass_a|workclass_b', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '<=50K'],
                               [6, 'workclass_a|workclass_c', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '<=50K'],
                               [10, 'workclass_a|workclass_c', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '<=50K'],
                               [24, 'workclass_a|workclass_c', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '<=50K'],
                               [56, 'workclass_a|workclass_c', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '<=50K'],
                               [10, 'workclass_a|workclass_c', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '<=50K']],
                              columns=columns)
    # train_data['age'].

    test_data = pd.DataFrame([[1, 'workclass_a|workclass_b', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '>50K'],
                               [1, 'workclass_a|workclass_b', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '>50K'],
                               [1, 'workclass_a|workclass_b', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '>50K'],
                               [1, 'workclass_a|workclass_b', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '>50K'],
                               [1, 'workclass_a|workclass_b', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '<=50K'],
                               [1, 'workclass_a|workclass_c', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '<=50K'],
                               [1, 'workclass_a|workclass_c', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '<=50K'],
                               [1, 'workclass_a|workclass_c', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '<=50K'],
                               [1, 'workclass_a|workclass_c', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '<=50K'],
                               [1, 'workclass_a|workclass_c', 1, 'a', 1, 'a', 'a', 'a', 'a', 'a', 1, 1, 1, 'a', '<=50K']],
                              columns=columns)

    train_data.to_csv('./train_data.csv', index=False)
    test_data.to_csv('./test_data.csv', index=False)
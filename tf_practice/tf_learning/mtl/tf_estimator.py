#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris=load_iris()
train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

print("train_x:", (train_y[:10]))

feature_columns = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


# 自定义模型函数
def my_model_fn(features, labels, mode, params):
    # 输入层,feature_columns对应Classifier(feature_columns=...)
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    # 隐藏层,hidden_units对应Classifier(unit=[10,10])，2个各含10节点的隐藏层
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # 输出层，n_classes对应3种鸢尾花
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # 预测
    predicted_classes = tf.argmax(logits, 1)  # 预测的结果中最大值即种类
    if mode == tf.contrib.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],  # 拼成列表[[3],[2]]格式
            'probabilities': tf.nn.softmax(logits),  # 把[-1.3,2.6,-0.9]规则化到0~1范围,表示可能性
            'logits': logits,  # [-1.3,2.6,-0.9]
        }
        return tf.contrib.estimator.EstimatorSpec(mode, predictions=predictions)

    # 损失函数
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # 训练
    if mode == tf.contrib.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)  # 用它优化损失函数，达到损失最少精度最高
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())  # 执行优化！
        return tf.contrib.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        # 评价
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')  # 计算精度
    metrics = {'accuracy': accuracy}  # 返回格式
    tf.summary.scalar('accuracy', accuracy[1])  # 仅为了后面图表统计使用
    if mode == tf.contrib.estimator.ModeKeys.EVAL:
        return tf.contrib.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


    #创建自定义分类器
classifier = tf.contrib.estimator.Estimator(
        model_fn=my_model_fn,
        params={
            'feature_columns': feature_columns,
            'hidden_units': [10, 10],
            'n_classes': 3,
        })

#针对训练的喂食函数
batch_size=10
def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size) #每次随机调整数据顺序
    return dataset.make_one_shot_iterator().get_next()

#开始训练
classifier.train(
    input_fn=lambda:train_input_fn(train_x, train_y, 100),
    steps=1000)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

def feat_map(raw_file):
    m = {}
    index = 0
    for line in open(raw_file, 'r'):
        line = line.split(',')
        line.pop(0)
        f_value = map(lambda x: tuple(x.split(':')), line)
        for fn, v in f_value:
            m.setdefault(fn, -1)
            if m.get(fn) < 0:
                m[fn] = index
                index += 1
    return m

def gen_tfrecords(raw_file, train_tfrecords_file, test_tfrecords_file, rate, m):
    train_writer = tf.python_io.TFRecordWriter(train_tfrecords_file)
    test_writer = tf.python_io.TFRecordWriter(test_tfrecords_file)
    train_line_num = test_line_num = 0
    for line in open(raw_file, "r"):
        line = line.split(",")
        if len(line)<=0:
            pass
        label = line.pop(0)
        label = [1, 0] if int(label) == 0 else [0, 1]

        feats = [0] * len(m)
        if len(line) <= 1:
            continue
        for feat in line:
            f_name, f_value = feat.split(":")
            feats[int(m[f_name])] = float(f_value)

        example = tf.train.Example(features=tf.train.Features(feature={
            "feats": tf.train.Feature(float_list=tf.train.FloatList(value=feats)),
            "label":tf.train.Feature(int64_list=tf.train.Int64List(value=label))
        }))
        if np.random.random() > rate:
            train_line_num += 1
            train_writer.write(example.SerializeToString())
        else:
            test_line_num += 1
            test_writer.write(example.SerializeToString())
    return train_line_num, test_line_num

input_data1 = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample"
train_tfrecords_data1 = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample_train_tfrecords1"
test_tfrecords_data1 = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample_test_tfrecords1"
input_data2 = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample"
train_tfrecords_data2 = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample_train_tfrecords2"
test_tfrecords_data2 = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample_test_tfrecords2"

# input_data = "/opt/meituan/cephfs/user/hadoop-ups/lichangqing03/have_child_806_01"
print("Start transform hive data to array......")
fni1 = feat_map(input_data1)
train_line_num1, test_line_num1 = gen_tfrecords(input_data1, train_tfrecords_data1, test_tfrecords_data1, 0.3, fni1)
fni2 = feat_map(input_data2)
train_line_num2, test_line_num2 = gen_tfrecords(input_data2, train_tfrecords_data2, test_tfrecords_data2, 0.3, fni2)
print("Transforming hive data to array success!")

epoch_num = 10
batch_size = 100

def read_tfrecords(tfrecords_file, batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={'label':tf.FixedLenFeature([2], tf.int64),
                                                 'feats':tf.FixedLenFeature([len(fni)], tf.float32)
                                                })
    feats = features['feats']
    label = features['label']
    feats_, label_ = tf.train.shuffle_batch([feats, label], batch_size=batch_size, capacity=batch_size, min_after_dequeue=1)
    return feats_, label_


# 1-child, 2-marry
x = tf.placeholder(tf.float32, [None, 35], name='x')

y_actual1 = tf.placeholder(tf.float32, shape=[None, 2], name='y_actual1')
y_actual2 = tf.placeholder(tf.float32, shape=[None, 2], name='y_actual2')

W_shared = tf.Variable(tf.random_normal(shape=[35, 100], mean=0, stddev=1), name='W_shared')
b_shared = tf.Variable(tf.zeros([100]), name='b_shared')

W1 = tf.Variable(tf.random_normal(shape=[100, 2], mean=0, stddev=1), name='W1')
b1 = tf.Variable(tf.zeros([2]), name='b1')  # 初始化偏置项b
W2 = tf.Variable(tf.random_normal(shape=[100, 2], mean=0, stddev=1), name='W2')
b2 = tf.Variable(tf.zeros([2]), name='b2')  # 初始化偏置项b

shared_out = tf.nn.tanh(tf.matmul(x, W_shared) + b_shared)

y_predict1 = tf.nn.softmax(tf.matmul(shared_out, W1) + b1, name='y_predict1')  # 加权变换并进行softmax回归，得到预测概率
y_predict2 = tf.nn.softmax(tf.matmul(shared_out, W2) + b2, name='y_predict2')  # 加权变换并进行softmax回归，得到预测概率

cross_entropy1 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_actual1, logits=y_predict1, dim=-1, name="cross_entropy1"))
train_step1 = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy1)  # 用梯度下降法使得残差最小
correct_prediction1 = tf.equal(tf.argmax(y_predict1, 1), tf.argmax(tf.nn.softmax(y_actual1, axis=1), 1))  # 在测试阶段，测试准确度计算
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, "float32"))  # 多个批次的准确度均值

cross_entropy2 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_actual2, logits=y_predict2, dim=-1, name="cross_entropy2"))
train_step2 = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy2)  # 用梯度下降法使得残差最小
correct_prediction2 = tf.equal(tf.argmax(y_predict2, 1), tf.argmax(tf.nn.softmax(y_actual2, axis=1), 1))  # 在测试阶段，测试准确度计算
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float32"))  # 多个批次的准确度均值


train_feats1, train_label1 = read_tfrecords(train_tfrecords_data1, batch_size)
test_feats1, test_label1 = read_tfrecords(test_tfrecords_data1, batch_size)
train_feats2, train_label2 = read_tfrecords(train_tfrecords_data2, batch_size)
test_feats2, test_label2 = read_tfrecords(test_tfrecords_data2, batch_size)

saver = tf.train.Saver()
save_dir = "./deep_child_model_save/"
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for epoch_i in range(epoch_num):
        for i in range(train_line_num//batch_size):
            train_feats_ = sess.run(train_feats)
            train_label_ = sess.run(train_label)
            # print(train_feats_)

            sess.run(train_step, feed_dict={x: train_feats_, y_actual: train_label_})

            test_feats_ = sess.run(test_feats)
            test_label_ = sess.run(test_label)
            print("Epoch %d Batch %d/%d test_accuracy: %.4f" % (epoch_i, i, train_line_num//batch_size, sess.run(accuracy, feed_dict={x: test_feats_, y_actual: test_label_})))

    saver.save(sess, save_dir+'model.ckpt')

    coord.request_stop()
    coord.join(threads=threads)
    print("Train Over!")

# 模型推理
test_feats, test_label = read_tfrecords(test_tfrecords_data, test_line_num)
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.import_meta_graph(save_dir + 'model.ckpt.meta')
    saver.restore(sess, "./deep_child_model_save/model.ckpt")
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name('x:0')
    result = graph.get_tensor_by_name('y_predict:0')

    test_feats_ = sess.run(test_feats)
    test_label_ = sess.run(test_label)
    infer = sess.run(result, feed_dict={x: test_feats_})

    with open("./predictions.txt", 'w') as f:
        for i in range(len(infer)):
            temp = str(1 if test_label_[i][1]==1 else 0) + ' ' + str(infer[i][1])
            f.write(temp + '\n')
    print("Save the prediction success!")

    coord.request_stop()
    coord.join(threads=threads)
    print("Inference Over!")
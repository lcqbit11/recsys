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
        label = [int(line.pop(0))]
        # label = [1, 0] if int(label) == 0 else [0, 1]

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

input_data = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample"
train_tfrecords_data = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample_train_tfrecords"
test_tfrecords_data = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample_test_tfrecords"

# input_data = "/opt/meituan/cephfs/user/hadoop-ups/lichangqing03/have_child_806_01"
print("Start transform hive data to array......")
fni = feat_map(input_data)
train_line_num, test_line_num = gen_tfrecords(input_data, train_tfrecords_data, test_tfrecords_data, 0.3, fni)
print("Transforming hive data to array success!")

epoch_num = 10
batch_size = 100

def read_tfrecords(tfrecords_file, batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={'label':tf.FixedLenFeature([1], tf.int64),
                                                 'feats':tf.FixedLenFeature([len(fni)], tf.float32)
                                                })
    feats = features['feats']
    label = features['label']
    feats_, label_ = tf.train.shuffle_batch([feats, label], batch_size=batch_size, capacity=batch_size, min_after_dequeue=1)
    return feats_, label_

x = tf.placeholder(tf.float32, [None, 35], name='x')
y_actual = tf.placeholder(tf.float32, shape=[None, 2], name='y_actual')
W = tf.Variable(tf.zeros([35,2]), name='W')        #初始化权值W
b = tf.Variable(tf.zeros([2]), name='b')            #初始化偏置项b
y_predict = tf.nn.softmax(tf.matmul(x,W) + b, name='y_predict')     #加权变换并进行softmax回归，得到预测概率
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(y_predict),reduction_indices=1))   #求交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)   #用梯度下降法使得残差最小

correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))   #在测试阶段，测试准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))                #多个批次的准确度均值

train_feats, train_label = read_tfrecords(train_tfrecords_data, batch_size)
test_feats, test_label = read_tfrecords(test_tfrecords_data, batch_size)

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
            train_label_ = [[1, 0] if int(item[0]) == 0 else [0, 1] for item in train_label_]

            # print(train_feats_)

            sess.run(train_step, feed_dict={x: train_feats_, y_actual: train_label_})

            test_feats_ = sess.run(test_feats)
            test_label_ = sess.run(test_label)
            test_label_ = [[1, 0] if int(item[0]) == 0 else [0, 1] for item in test_label_]
            print("Epoch %d Batch %d/%d test_accuracy: %.4f" % (epoch_i, i, train_line_num//batch_size, sess.run(accuracy, feed_dict={x: test_feats_, y_actual: test_label_})))

    saver.save(sess, save_dir+'model_v2.ckpt')

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

    saver = tf.train.import_meta_graph(save_dir + 'model_v2.ckpt.meta')
    saver.restore(sess, "./deep_child_model_save/model_v2.ckpt")
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name('x:0')
    result = graph.get_tensor_by_name('y_predict:0')

    test_feats_ = sess.run(test_feats)
    test_label_ = sess.run(test_label)
    test_label_ = [[1, 0] if int(item[0]) == 0 else [0, 1] for item in test_label_]
    infer = sess.run(result, feed_dict={x: test_feats_})

    with open("./predictions.txt", 'w') as f:
        for i in range(len(infer)):
            temp = str(1 if test_label_[i][1]==1 else 0) + ' ' + str(infer[i][1])
            f.write(temp + '\n')
    print("Save the prediction success!")

    coord.request_stop()
    coord.join(threads=threads)
    print("Inference Over!")
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
#         label = [1, 0] if int(label) == 0 else [0, 1]

#         values = [0] * 35
#         if len(line) <= 1:
#             continue
#         for feat in line:
#             f_name, f_value = feat.split(":")
        kvs = map(lambda x: tuple(x.split(':')), line)
        idxs = []
        values = []
        for k, v in kvs:
            idxs.append(float(m[k]))
            values.append(float(v))

        example = tf.train.Example(features=tf.train.Features(feature={
            "idxs": tf.train.Feature(float_list=tf.train.FloatList(value=idxs)),
            "values": tf.train.Feature(float_list=tf.train.FloatList(value=values)),
            "labels":tf.train.Feature(int64_list=tf.train.Int64List(value=label))
        }))
        if np.random.random() > rate:
            train_line_num += 1
            train_writer.write(example.SerializeToString())
        else:
            test_line_num += 1
            test_writer.write(example.SerializeToString())
    return train_line_num, test_line_num


def read_tfrecords(tfrecords_file, batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={'labels':tf.FixedLenFeature([1], tf.int64),
                                                 'idxs':tf.VarLenFeature(tf.float32),
                                                 'values':tf.VarLenFeature(tf.float32)
                                                })
    idxs = features['idxs']
    values = features['values']
    labels = features['labels']
#     values_, label_ = tf.train.shuffle_batch([values, labels], batch_size=batch_size, capacity=batch_size, min_after_dequeue=1)

    idxs_, values_, label_ = tf.train.shuffle_batch([idxs, values, labels], batch_size=batch_size, capacity=batch_size, min_after_dequeue=1)
    return idxs_, values_, label_
#     return values_, label_

input_data = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample"
train_tfrecords_data = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample_train_tfrecords"
test_tfrecords_data = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample_test_tfrecords"

# input_data = "/opt/meituan/cephfs/user/hadoop-ups/lichangqing03/have_child_806_01"
print("Start transform hive data to array......")
fni = feat_map(input_data)
print("feature numbers=%d" % len(fni))
train_line_num, test_line_num = gen_tfrecords(input_data, train_tfrecords_data, test_tfrecords_data, 0.3, fni)
print("Transforming hive data to array success!")

epoch_num = 1
batch_size = 100

x = tf.sparse_placeholder(tf.float32, [None, 35], name='input_x')

# tf.add_to_collection('xxx', x)

y_actual = tf.placeholder(tf.float32, shape=[None, 2], name='y_actual')
W = tf.Variable(tf.zeros([35,2]), name='W')        #初始化权值W
b = tf.Variable(tf.zeros([2]), name='b')            #初始化偏置项b
y_predict = tf.nn.softmax(tf.sparse_tensor_dense_matmul(x,W) + b, name='y_predict')     #加权变换并进行softmax回归，得到预测概率
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(y_predict),reduction_indices=1))   #求交叉熵
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)   #用梯度下降法使得残差最小
train_step = tf.train.AdamOptimizer(learning_rate=0.001,)

reg_l2 = tf.contrib.layers.l2_regularizer(0.1)
x = tf.nn.local_response_normalization()


correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))   #在测试阶段，测试准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))                #多个批次的准确度均值

train_idxs, train_values, train_labels = read_tfrecords(train_tfrecords_data, batch_size)
test_idxs, test_values, test_labels = read_tfrecords(test_tfrecords_data, batch_size)

saver = tf.train.Saver()
save_dir = "./deep_child_model_save/"
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for epoch_i in range(epoch_num):
        for i in range(train_line_num//batch_size):
            train_idxs_ = sess.run(train_idxs)
            train_values_ = sess.run(train_values)
            train_labels_ = sess.run(train_labels)
            train_labels_ = [[1, 0] if int(item[0]) == 0 else [0, 1] for item in train_labels_]
            train_values_new_ = tf.SparseTensorValue(indices=train_values_.indices, values=train_values_.values, dense_shape=[batch_size, 35])
            sess.run(train_step, feed_dict={x: train_values_new_, y_actual: train_labels_})

            test_idxs_ = sess.run(test_idxs)
            test_values_ = sess.run(test_values)
            test_labels_ = sess.run(test_labels)
            test_labels_ = [[1, 0] if int(item[0]) == 0 else [0, 1] for item in test_labels_]
            test_values_new_ = tf.SparseTensorValue(indices=test_values_.indices, values=test_values_.values,
                                                     dense_shape=[batch_size, 35])
            print("Epoch %d Batch %d/%d test_accuracy: %.4f" % (epoch_i, i, train_line_num//batch_size, sess.run(accuracy, feed_dict={x: test_values_new_, y_actual: test_labels_})))

    saver.save(sess, save_dir+'model.ckpt')
    coord.request_stop()
    coord.join(threads=threads)
    print("Train Over!")

# 模型推理
# test_idxs, test_values, test_labels = read_tfrecords(test_tfrecords_data, test_line_num)
# print("test_idxs", test_idxs)
graph = tf.Graph()
# tf.reset_default_graph()
saver = tf.train.import_meta_graph(save_dir + 'model.ckpt.meta')
with tf.Session(graph=graph) as sess:
    init_op = tf.global_variables_initializer()
    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver.restore(sess, "./deep_child_model_save/model.ckpt")
    # graph = tf.get_default_graph()

    # for op in graph.get_operations():
    #     print(op.name)

    x_indices = graph.get_tensor_by_name("input_x/indices:0")
    x_values = graph.get_tensor_by_name("input_x/values:0")
    x_shape = graph.get_tensor_by_name("input_x/shape:0")

    result = graph.get_tensor_by_name('y_predict:0')

    print("result:", result)

    # print("x_indices:", x_indi    ces)
    # print("x_values:", x_values)
    # print("x_shape:", x_shape)
    # test_idxs_ = sess.run(test_idxs)
    test_values_ = sess.run(test_values)
    test_labels_ = sess.run(test_labels)
    test_labels_ = [[1, 0] if int(item[0]) == 0 else [0, 1] for item in test_labels_]
    test_values_new_ = tf.SparseTensorValue(indices=test_values_.indices, values=test_values_.values,
                                            dense_shape=[np.shape(test_labels_)[0], 35])
    infer = sess.run(result, feed_dict={x: test_values_new_})
    # infer = sess.run(result, feed_dict={x_indices: test_values_.indices, x_values: test_values_.values, x_shape: [np.shape(test_labels_)[0], 35]})

    with open("./predictions.txt", 'w') as f:
        for i in range(len(infer)):
            temp = str(1 if test_labels_[i][1]==1 else 0) + ' ' + str(infer[i][1])
            f.write(temp + '\n')
    print("Save the prediction success!")

    coord.request_stop()
    coord.join(threads=threads)
    print("Inference Over!")
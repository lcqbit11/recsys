#!/user/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


def hiveData2ndarray(input_libsvm_file, num_feat):
    reader = open(input_libsvm_file)
    train_data_array = []
    target_data_array = []
    m = {}
    index = 0
    for line in reader:
        train_data = []

        line = line.strip().split(',')
        label = line.pop(0)
        target_data_array.append(label)

        line = map(lambda x: tuple(x.split(":")), line)
        train_data = [0] * num_feat
        for i, v in line:
            m.setdefault(i, -1)
            if m.get(i) < 0:
                m[i] = index
                index += 1
            if m[i] < num_feat:
                train_data[m[i]] = v
        train_data_array.append(train_data)
    return train_data_array, target_data_array

def shuffle_data(train_data_array, target_data_array):
    cur_state = np.random.get_state()
    np.random.shuffle(train_data_array)
    np.random.set_state(cur_state)
    np.random.shuffle(target_data_array)

# 取得batch
def get_batch(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start+batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]

input_data = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample"
number_feat = 35
print("Start transform hive data to array......")
train_data_array, target_data_array = hiveData2ndarray(input_data, number_feat)
train_data_array = np.array(train_data_array)
target_data_array = np.array(target_data_array)
print("Transforming hive data to array success!")

shuffle_data(train_data_array, target_data_array)
train_X, test_X, train_y, test_y = train_test_split(train_data_array, target_data_array, test_size=0.3, random_state=0)

train_y = [[1, 0] if label=='0' else [0, 1] for label in train_y]
test_y = [[1, 0] if label=='0' else [0, 1] for label in test_y]

x = tf.placeholder(tf.float32, [None, 35], name='x')
y_actual = tf.placeholder(tf.float32, shape=[None, 2], name='y_actual')
W = tf.Variable(tf.zeros([35, 2]), name='W')  # 初始化权值W
b = tf.Variable(tf.zeros([2]), name='b')  # 初始化偏置项b
y_predict = tf.nn.softmax(tf.matmul(x, W) + b, name='y_predict')  # 加权变换并进行softmax回归，得到预测概率
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_predict), reduction_indices=1))  # 求交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  # 用梯度下降法使得残差最小

correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))  # 在测试阶段，测试准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 多个批次的准确度均值

epoch_num = 10
batch_size = 128
saver = tf.train.Saver()
save_dir = "./deep_child_model_save/"
init = tf.global_variables_initializer()
# init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for epoch_i in range(epoch_num):
        shuffle_data(train_X, train_y)
        shuffle_data(test_X, test_y)
        batch_train = get_batch(train_X, train_y, batch_size)
        batch_test = get_batch(test_X, test_y, batch_size)
        for i in range(len(train_X) // batch_size):
            try:
                batch_train_xs, batch_train_ys = next(batch_train)  # 按批次训练，每批100行数据
            except StopIteration:
                sys.exit
            sess.run(train_step, feed_dict={x: batch_train_xs, y_actual: batch_train_ys})  # 执行训练
            if (i % 5 == 0):  # 每训练100次，测试一次
                try:
                    batch_test_xs, batch_test_ys = next(batch_test)
                except StopIteration:
                    sys.exit
                print("Epoch %d Batch %d/%d test_accuracy: %.4f" % (epoch_i, i, len(train_X) // batch_size,
                                                                    sess.run(accuracy, feed_dict={x: batch_test_xs,
                                                                                                  y_actual: batch_test_ys})))

    saver.save(sess, save_dir+'model.ckpt')

# 多目标预测
# ckpt = tf.train.get_checkpoint_state(save_dir)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(save_dir + 'model.ckpt.meta')

    saver.restore(sess, "./deep_child_model_save/model.ckpt")
    graph = tf.get_default_graph()
    # sess.run(tf.global_variables_initializer())

    x = graph.get_tensor_by_name('x:0')
    result = graph.get_tensor_by_name('y_predict:0')

    infer = sess.run(result, feed_dict={x: test_X})
    print("inference type:", (infer[:]))
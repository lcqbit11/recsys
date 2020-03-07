#!/user/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# input_marry_data = "/opt/meituan/cephfs/user/hadoop-ups/lichangqing03/marry_data"
input_child_data = "./child_app_feat"
fn_fid_map = './fn_fid_map.txt'
number_feat = 1973 # common feat num


fname_id = {}
with open(fn_fid_map, 'r') as f:
    for line in f:
        line = line.split(' ')
        fname_id.setdefault(int(line[0]), int(line[1]))

def hiveData2ndarray(input_file, num_feat, m):
    reader = open(input_file, 'rb')
    train_data_array = []
    target_data_array = []
    for line in reader:
        line = line.strip().split(b'\t')
        label = int(line[1])
        label = [1, 0] if label == 0  else [0, 1]
        target_data_array.append(label)
        line = line[2].strip().split(b',')

        train_data = [0] * num_feat

        for l in line:
            l = l.split(b':')
            train_data[int(m[int(l[0])])] = int(l[1])
        # print(train_data)
    print(target_data_array)
    return train_data_array, target_data_array


def shuffle_data(train_data_array, target_data_array):
    cur_state = np.random.get_state()
    np.random.shuffle(train_data_array)
    np.random.set_state(cur_state)
    np.random.shuffle(target_data_array)


# 取得batch
def get_batch(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]

print("Start transform hive data to array......")



train_child_array, target_child_array = hiveData2ndarray(input_child_data, number_feat, fname_id)
print("train_child_array:", np.shape(train_child_array))
train_child_array = np.array(train_child_array, dtype=np.float32)
target_child_array = np.array(target_child_array, dtype=np.float32)

print("Transforming hive data to array success!")

shuffle_data(train_child_array, target_child_array)
print(train_child_array)
train_cX, test_cX, train_cy, test_cy = train_test_split(train_child_array, target_child_array, test_size=0.3, random_state=0)

train_cy = [[1, 0] if label=='0' else [0, 1] for label in train_cy]
test_cy = [[1, 0] if label=='0' else [0, 1] for label in test_cy]

# 1-child, 2-marry
x = tf.placeholder(tf.float32, [None, number_feat], name='input')

y_actual1 = tf.placeholder(tf.float32, shape=[None, 2], name='label1')

init_random = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=0, dtype=tf.float32)
init_zeros = tf.zeros_initializer(dtype=tf.float32)
reg_l2 = tf.contrib.layers.l2_regularizer(0.1)
learning_rate = 0.01
dropout_rate = 0
use_bn = False

shared_out = tf.layers.dense(x, 100, activation=tf.nn.tanh, kernel_initializer=init_random, kernel_regularizer=reg_l2)
if dropout_rate > 0:
    shared_out = tf.layers.dropout(shared_out, dropout_rate)
if use_bn:
    shared_out = tf.layers.batch_normalization(shared_out, axis=1, training=True)

y_predict1 = tf.layers.dense(shared_out, 2, kernel_initializer=init_random)

cross_entropy1 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_actual1, logits=y_predict1, name='cross_entropy1'))
train_step1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy1)

correct_prediction1 = tf.equal(tf.argmax(y_actual1, axis=1), tf.argmax(tf.nn.softmax(y_predict1), axis=1))
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

epoch_num = 20
batch_size = 512
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch_i in range(epoch_num):
        shuffle_data(train_cX, train_cy)
        shuffle_data(test_cX, test_cy)
        #         shuffle_data(train_mX, train_my)
        #         shuffle_data(test_mX, test_my)
        batch_train1 = get_batch(train_cX, train_cy, batch_size)
        batch_test1 = get_batch(test_cX, test_cy, batch_size)
        #         batch_train2 = get_batch(train_mX, train_my, batch_size)
        #         batch_test2 = get_batch(test_mX, test_my, batch_size)
        c_index = m_index = 0
        c_all_batch = len(train_cX) // batch_size
        #         m_all_batch = len(train_mX) // batch_size
        #         c_all_batch_ratio = 1.0 * c_all_batch / (c_all_batch+m_all_batch)
        for i in range(c_all_batch):
            try:
                batch_train_xs, batch_train_ys = next(batch_train1)  # 按批次训练，每批100行数据
            except StopIteration:
                sys.exit
            sess.run(train_step1, feed_dict={x: batch_train_xs, y_actual1: batch_train_ys})  # 执行训练
            try:
                batch_test_xs, batch_test_ys = next(batch_test1)
            except StopIteration:
                sys.exit
            acc1, loss = sess.run([accuracy1, cross_entropy1], feed_dict={x: batch_test_xs, y_actual1: batch_test_ys})
            print("Child: Epoch %d Train Batch %d/%d test_accuracy: %.4f, loss: %.4f" % (
            epoch_i, c_index, len(train_cX) // batch_size, acc1, loss))
    print("Train success!")


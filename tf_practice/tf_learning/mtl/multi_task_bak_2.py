#!/user/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import  sys


def hiveData2ndarray(input_file, num_feat, m):
    reader = open(input_file, 'rb')
    train_data_array = []
    target_data_array = []
    #     m = {}
    index = 0
    for line in reader:
        train_data = []

        line = line.strip().split(b"\t")
        label = line[1]
        target_data_array.append(label)
        line = line[2].strip().split(b',')

        line = map(lambda x: tuple(x.split(b":")), line)
        train_data = [0] * num_feat

        for i, v in line:
            m.setdefault(i, -1)
            if m.get(i) < 0:
                m[i] = index
                index += 1
            train_data[m[i]] = v
        train_data_array.append(train_data[:num_feat])
    print(input_file, "m:", len(m))
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

# prepare train data
input_marry_data = "/Users/lcq-mac/pycharm_projects/algorithms/tianchi_recommendation/marry_data"
input_child_data = "/Users/lcq-mac/pycharm_projects/algorithms/tianchi_recommendation/child_data"

print("Start transform hive data to array......")
number_marry_feat = 6859 # marry feat num
number_child_feat = 6969 # child feat num
number_feat = 7302 # common feat num

fname_id = {}
train_child_array, target_child_array = hiveData2ndarray(input_child_data, 7302, fname_id)
train_marry_array, target_marry_array = hiveData2ndarray(input_marry_data, 7302, fname_id)

train_marry_array = np.array(train_marry_array, dtype=np.float32)
target_marry_array = np.array(target_marry_array, dtype=np.float32)
train_child_array = np.array(train_child_array, dtype=np.float32)
target_child_array = np.array(target_child_array, dtype=np.float32)

print("Transforming hive data to array success!")

shuffle_data(train_child_array, target_child_array)
shuffle_data(train_marry_array, target_marry_array)

train_cX, test_cX, train_cy, test_cy = train_test_split(train_child_array, target_child_array, test_size=0.3, random_state=0)
train_mX, test_mX, train_my, test_my = train_test_split(train_marry_array, target_marry_array, test_size=0.3, random_state=0)

train_cy = [[1, 0] if label=='0' else [0, 1] for label in train_cy]
test_cy = [[1, 0] if label=='0' else [0, 1] for label in test_cy]
train_my = [[1, 0] if label=='0' else [0, 1] for label in train_my]
test_my = [[1, 0] if label=='0' else [0, 1] for label in test_my]

# init op
init_random = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=0, dtype=tf.float32)
init_zeros = tf.zeros_initializer(dtype=tf.float32)
reg_l2 = tf.contrib.layers.l2_regularizer(0.1)
learning_rate = 0.001
dropout_rate = 0.5
training = True

# 1-child, 2-marry
x = tf.placeholder(tf.float32, [None, number_feat], name='input')

y_actual1 = tf.placeholder(tf.float32, shape=[None, 2], name='y_actual1')
y_actual2 = tf.placeholder(tf.float32, shape=[None, 2], name='y_actual2')

# W_shared = tf.get_variable(name='W_shared', shape=[number_feat, 100], dtype=tf.float32, initializer=init_random, regularizer=reg_l2)
# b_shared = tf.get_variable(name='b_shared', shape=[100], dtype=tf.float32, initializer=init_zeros, regularizer=reg_l2)
#
# W1 = tf.get_variable(name='W1', shape=[100, 2], dtype=tf.float32, initializer=init_random, regularizer=reg_l2)
# b1 = tf.get_variable(name='b1', shape=[2], dtype=tf.float32, initializer=init_zeros, regularizer=reg_l2)
# W2 = tf.get_variable(name='W2', shape=[100, 2], dtype=tf.float32, initializer=init_random, regularizer=reg_l2)
# b2 = tf.get_variable(name='b2', shape=[2], dtype=tf.float32, initializer=init_zeros, regularizer=reg_l2)
#
# shared_out = tf.nn.sigmoid(tf.matmul(x, W_shared) + b_shared)
# y_predict1 = tf.matmul(shared_out, W1) + b1
# y_predict2 = tf.matmul(shared_out, W2) + b2
#
# cross_entropy1 = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_actual1, logits=y_predict1, dim=-1, name="cross_entropy1"))
# # cross_entropy1 = tf.reduce_mean(-tf.reduce_sum(y_actual1*tf.log(y_predict1),reduction_indices=1))   #求交叉熵
# train_step1 = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy1)  # 用梯度下降法使得残差最小
# correct_prediction1 = tf.equal(tf.argmax(y_predict1, 1), tf.argmax(tf.nn.softmax(y_actual1, axis=1), 1))  # 在测试阶段，测试准确度计算
# accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, "float32"))  # 多个批次的准确度均值
#
# cross_entropy2 = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_actual2, logits=y_predict2, dim=-1, name="cross_entropy2"))
# # cross_entropy2 = tf.reduce_mean(-tf.reduce_sum(y_actual2*tf.log(y_predict2),reduction_indices=1))   #求交叉熵
# train_step2 = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy2)  # 用梯度下降法使得残差最小
# correct_prediction2 = tf.equal(tf.argmax(y_predict2, 1), tf.argmax(tf.nn.softmax(y_actual2, axis=1), 1))  # 在测试阶段，测试准确度计算
# accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float32"))  # 多个批次的准确度均值


shared_out = tf.layers.dense(x, 100, activation=tf.nn.relu6, kernel_initializer=init_random, kernel_regularizer=reg_l2)
if dropout_rate>0:
    shared_out = tf.layers.dropout(shared_out, dropout_rate)
shared_out = tf.layers.batch_normalization(shared_out, axis=1, training=training)

y_predict1 = tf.layers.dense(shared_out, 2, kernel_initializer=init_random)
y_predict2 = tf.layers.dense(shared_out, 2, kernel_initializer=init_random)

cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_actual1, logits=y_predict1, dim=-1, name='cross_entropy1'))
train_step1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy1)

cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_actual2, logits=y_predict2, dim=-1, name='cross_entropy2'))
train_step2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy2)

correct_prediction1 = tf.equal(tf.argmax(y_actual1, axis=1), tf.argmax(tf.nn.softmax(y_predict1), axis=1))
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, "float32"))

correct_prediction2 = tf.equal(tf.argmax(y_actual2, axis=1), tf.argmax(tf.nn.softmax(y_predict2), axis=1))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float32"))


epoch_num = 500
batch_size = 218
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch_i in range(epoch_num):
        shuffle_data(train_cX, train_cy)
        shuffle_data(test_cX, test_cy)
        shuffle_data(train_mX, train_my)
        shuffle_data(test_mX, test_my)
        batch_train1 = get_batch(train_cX, train_cy, batch_size)
        batch_test1 = get_batch(test_cX, test_cy, batch_size)
        batch_train2 = get_batch(train_mX, train_my, batch_size)
        batch_test2 = get_batch(test_mX, test_my, batch_size)
        c_index = m_index = 0
        c_all_batch = len(train_cX) // batch_size
        m_all_batch = len(train_mX) // batch_size
        c_all_batch_ratio = 1.0 * c_all_batch / (c_all_batch + m_all_batch)
        for i in range(c_all_batch + m_all_batch):
            if c_index < c_all_batch and m_index < m_all_batch:
                if np.random.rand() < c_all_batch_ratio:
                    c_index += 1
                    try:
                        batch_train_xs, batch_train_ys = next(batch_train1)  # 按批次训练，每批100行数据
                    except StopIteration:
                        sys.exit
                    sess.run(train_step1, feed_dict={x: batch_train_xs, y_actual1: batch_train_ys})  # 执行训练
                    try:
                        batch_test_xs, batch_test_ys = next(batch_test1)
                    except StopIteration:
                        sys.exit
                    print("Child: Epoch %d Train Batch %d/%d test_accuracy: %.4f" % (
                    epoch_i, c_index, len(train_cX) // batch_size,
                    sess.run(accuracy1, feed_dict={x: batch_test_xs, y_actual1: batch_test_ys})))
                else:
                    m_index += 1
                    try:
                        batch_train_xs, batch_train_ys = next(batch_train2)  # 按批次训练，每批100行数据
                    except StopIteration:
                        sys.exit
                    sess.run(train_step2, feed_dict={x: batch_train_xs, y_actual2: batch_train_ys})  # 执行训练
                    try:
                        batch_test_xs, batch_test_ys = next(batch_test2)
                    except StopIteration:
                        sys.exit
                    print("Marry: Epoch %d Train Batch %d/%d (easy observation) test_accuracy: %.4f" % (
                    epoch_i, m_index, len(train_mX) // batch_size,
                    sess.run(accuracy2, feed_dict={x: batch_test_xs, y_actual2: batch_test_ys})))

            if c_index >= c_all_batch and m_index < m_all_batch:
                m_index += 1
                try:
                    batch_train_xs, batch_train_ys = next(batch_train2)  # 按批次训练，每批100行数据
                except StopIteration:
                    sys.exit
                sess.run(train_step2, feed_dict={x: batch_train_xs, y_actual2: batch_train_ys})  # 执行训练
                try:
                    batch_test_xs, batch_test_ys = next(batch_test2)
                except StopIteration:
                    sys.exit
                print("Marry: Epoch %d Train Batch %d/%d (easy observation) test_accuracy: %.4f" % (
                epoch_i, m_index, len(train_mX) // batch_size,
                sess.run(accuracy2, feed_dict={x: batch_test_xs, y_actual2: batch_test_ys})))

            if m_index >= m_all_batch and c_index < c_all_batch:
                c_index += 1
                try:
                    batch_train_xs, batch_train_ys = next(batch_train1)  # 按批次训练，每批100行数据
                except StopIteration:
                    sys.exit
                sess.run(train_step1, feed_dict={x: batch_train_xs, y_actual1: batch_train_ys})  # 执行训练
                try:
                    batch_test_xs, batch_test_ys = next(batch_test1)
                except StopIteration:
                    sys.exit
                print("Child: Epoch %d Train Batch %d/%d test_accuracy: %.4f" % (
                epoch_i, c_index, len(train_cX) // batch_size,
                sess.run(accuracy1, feed_dict={x: batch_test_xs, y_actual1: batch_test_ys})))



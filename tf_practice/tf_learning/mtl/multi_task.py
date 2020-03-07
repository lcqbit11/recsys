#!/user/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
# super params
learning_rate = 0.001
rate = 0.5
training = True
epoch_num = 100
batch_size = 256
# 输出日志保存的路径
log_dir = '/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/'

# 1-child, 2-marry
# with tf.name_scope('input'):
x = tf.placeholder(tf.float32, [None, number_feat], name='x_input')
y_actual1 = tf.placeholder(tf.float32, shape=[None, 2], name='label1')
y_actual2 = tf.placeholder(tf.float32, shape=[None, 2], name='label2')

# with tf.name_scope("layer"):
shared_out = tf.layers.dense(x, 100, activation=tf.nn.relu6, kernel_initializer=init_random, kernel_regularizer=reg_l2, name="shared_out")
if rate>0:
    shared_out = tf.nn.dropout (shared_out, rate=rate, name='dropout')
shared_out = tf.layers.batch_normalization(shared_out, axis=1, training=training, name='bn_normalization')

y_predict1 = tf.layers.dense(shared_out, 2, kernel_initializer=init_random, name='y_predict1')
y_predict2 = tf.layers.dense(shared_out, 2, kernel_initializer=init_random, name='y_predict2')

# with tf.name_scope("loss1"):
cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_actual1, logits=y_predict1, dim=-1, name='cross_entropy1'))
tf.summary.scalar('loss1', cross_entropy1)
# with tf.name_scope("train1"):
train_step1 = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy1)  # 用梯度下降法使得残差最小
# with tf.name_scope("accuracy1"):
correct_prediction1 = tf.equal(tf.argmax(y_actual1, axis=1), tf.argmax(tf.nn.softmax(y_predict1), axis=1))
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, "float32"))
tf.summary.scalar("accuracy1", accuracy1)
# global_step1 = tf.Variable(0, name="global_step", trainable=False)
# optimizer1 = tf.train.GradientDescentOptimizer(learning_rate)
# gradients1 = optimizer1.compute_gradients(cross_entropy1)
# train_op1 = optimizer1.apply_gradients(gradients1, global_step1)

# with tf.name_scope("loss2"):
cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_actual2, logits=y_predict2, dim=-1, name='cross_entropy2'))
# tf.summary.scalar('loss2', cross_entropy2)
# with tf.name_scope("train2"):
train_step2 = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy2)  # 用梯度下降法使得残差最小
# with tf.name_scope("accuracy2"):
correct_prediction2 = tf.equal(tf.argmax(y_actual2, axis=1), tf.argmax(tf.nn.softmax(y_predict2), axis=1))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float32"))
tf.summary.scalar("accuracy2", accuracy2)

# global_step2 = tf.Variable(0, name="global_step", trainable=False)
# optimizer2 = tf.train.GradientDescentOptimizer(learning_rate)
# gradients2 = optimizer2.compute_gradients(cross_entropy2)
# train_op2 = optimizer2.apply_gradients(gradients2, global_step2)

# summaries 合并
# merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test', )

    acc1_list = []
    acc2_list = []
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
                    # summary, _ = sess.run([merged, train_step1], feed_dict={x: batch_train_xs, y_actual1: batch_train_ys})  # 执行训练
                    # train_writer.add_summary(summary, i)
                    try:
                        batch_test_xs, batch_test_ys = next(batch_test1)
                    except StopIteration:
                        sys.exit
                    acc1 = sess.run(accuracy1, feed_dict={x: batch_test_xs, y_actual1: batch_test_ys})
                    # test_writer.add_summary(summary, i)
                    acc1_list.append(acc1)
                    print("Child: Epoch %d Train Batch %d/%d test_accuracy: %.4f" % (
                    epoch_i, c_index, len(train_cX) // batch_size, acc1))
                else:
                    m_index += 1
                    try:
                        batch_train_xs, batch_train_ys = next(batch_train2)  # 按批次训练，每批100行数据
                    except StopIteration:
                        sys.exit
                    sess.run(train_step2, feed_dict={x: batch_train_xs, y_actual2: batch_train_ys})  # 执行训练

                    # summary, _ = sess.run([merged, train_step2], feed_dict={x: batch_train_xs, y_actual2: batch_train_ys})  # 执行训练
                    # train_writer.add_summary(summary, i)
                    try:
                        batch_test_xs, batch_test_ys = next(batch_test2)
                    except StopIteration:
                        sys.exit
                    acc2 = sess.run(accuracy2, feed_dict={x: batch_test_xs, y_actual2: batch_test_ys})

                    # summary, acc2 = sess.run([merged, accuracy2], feed_dict={x: batch_test_xs, y_actual2: batch_test_ys})
                    # test_writer.add_summary(summary, i)
                    acc2_list.append(acc2)
                    print("Marry: Epoch %d Train Batch %d/%d (easy observation) test_accuracy: %.4f" % (
                    epoch_i, m_index, len(train_mX) // batch_size, acc2))

            if c_index >= c_all_batch and m_index < m_all_batch:
                m_index += 1
                try:
                    batch_train_xs, batch_train_ys = next(batch_train2)  # 按批次训练，每批100行数据
                except StopIteration:
                    sys.exit
                sess.run(train_step2,
                                      feed_dict={x: batch_train_xs, y_actual2: batch_train_ys})  # 执行训练

                # summary, _ = sess.run([merged, train_step2],
                #                       feed_dict={x: batch_train_xs, y_actual2: batch_train_ys})  # 执行训练
                # train_writer.add_summary(summary, i)
                try:
                    batch_test_xs, batch_test_ys = next(batch_test2)
                except StopIteration:
                    sys.exit
                acc2 = sess.run(accuracy2, feed_dict={x: batch_test_xs, y_actual2: batch_test_ys})

                # summary, acc2 = sess.run([merged, accuracy2], feed_dict={x: batch_test_xs, y_actual2: batch_test_ys})
                # test_writer.add_summary(summary, i)
                acc2_list.append(acc2)
                print("Marry: Epoch %d Train Batch %d/%d (easy observation) test_accuracy: %.4f" % (
                epoch_i, m_index, len(train_mX) // batch_size, acc2))

            if m_index >= m_all_batch and c_index < c_all_batch:
                c_index += 1
                try:
                    batch_train_xs, batch_train_ys = next(batch_train1)  # 按批次训练，每批100行数据
                except StopIteration:
                    sys.exit
                # summary, _ = sess.run([merged, train_step1],
                #                       feed_dict={x: batch_train_xs, y_actual1: batch_train_ys})  # 执行训练
                # train_writer.add_summary(summary, i)
                try:
                    batch_test_xs, batch_test_ys = next(batch_test1)
                except StopIteration:
                    sys.exit
                acc1 = sess.run(accuracy1, feed_dict={x: batch_test_xs, y_actual1: batch_test_ys})
                # test_writer.add_summary(summary, i)
                acc1_list.append(acc1)
                print("Child: Epoch %d Train Batch %d/%d test_accuracy: %.4f" % (
                epoch_i, c_index, len(train_cX) // batch_size, acc1))

    plt.figure(1, figsize=[12, 5])

    plt.subplot(1, 2, 1)
    plt.plot(acc1_list)
    plt.title("Child acc optimization cure")
    plt.xlabel("batch index")
    plt.ylabel("Child Acc")

    plt.subplot(1, 2, 2)
    plt.plot(acc2_list)
    plt.title("Marry acc optimization cure")
    plt.xlabel("batch index")
    plt.ylabel("Marry Acc")

    plt.show()
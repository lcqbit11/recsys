#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

epoch_num = 10
batch_size = 100
train_tfrecords_data = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample_train_tfrecords"
test_tfrecords_data = "/Users/lcq-mac/pycharm_projects/algorithms/tf_practice/tf_learning/mtl/child_part_sample_test_tfrecords"

# train_tfrecords_data = '/opt/meituan/cephfs/user/hadoop-ups/lichangqing03/from_spark/child_tfrecords_train'
# test_tfrecords_data = '/opt/meituan/cephfs/user/hadoop-ups/lichangqing03/from_spark/child_tfrecords_test'

def read_tfrecords(tfrecords_file, batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'label':tf.FixedLenFeature([2], tf.int64),'feats':tf.FixedLenFeature([35], tf.float32)})
    feats = features['feats']
    label = features['label']
    feats_, label_ = tf.train.shuffle_batch([feats, label], batch_size=batch_size, capacity=batch_size, min_after_dequeue=1)
    print("label_:", label_)
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

train_sample_nums = 0
for record in tf.python_io.tf_record_iterator(train_tfrecords_data):
    print("record:", record)
    print("record type:", type(record))
    print("record shape:", np.shape(record))
    train_sample_nums += 1
test_sample_nums = 0
for _ in tf.python_io.tf_record_iterator(train_tfrecords_data):
    test_sample_nums += 1

# 模型训练
saver = tf.train.Saver()
save_dir = "./deep_child_model_save/"
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for epoch_i in range(epoch_num):
        for i in range(train_sample_nums//batch_size):
            train_feats_ = sess.run(train_feats)
            train_label_ = sess.run(train_label)
            # print(train_feats_)

            sess.run(train_step, feed_dict={x: train_feats_, y_actual: train_label_})

            test_feats_ = sess.run(test_feats)
            test_label_ = sess.run(test_label)
            test_label_ = [[1, 0] if int(item[0]) == 0 else [0, 1] for item in test_label_]
            print("Epoch %d Batch %d/%d test_accuracy: %.4f" % (epoch_i, i, train_sample_nums//batch_size, sess.run(accuracy, feed_dict={x: test_feats_, y_actual: test_label_})))

    saver.save(sess, save_dir+'model.ckpt')

    coord.request_stop()
    coord.join(threads=threads)
    print("Train Over!")

# 模型推理
test_feats, test_label = read_tfrecords(test_tfrecords_data, test_sample_nums)
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
    print("test_label_:", test_label_)
    print("test_label_:", np.shape(test_label_))

    infer = sess.run(result, feed_dict={x: test_feats_})

    with open("./predictions.txt", 'w') as f:
        for i in range(len(infer)):
            temp = str(1 if test_label_[i][1]==1 else 0) + ' ' + str(infer[i][1])
            f.write(temp + '\n')
    print("Save the prediction success!")

    coord.request_stop()
    coord.join(threads=threads)
    print("Inference Over!")
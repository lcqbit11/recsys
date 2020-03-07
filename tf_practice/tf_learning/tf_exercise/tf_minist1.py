#!/user/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data


mnist = input_data.read_data_sets("./MNIST_data1", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y_actual = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))        #初始化权值W
b = tf.Variable(tf.zeros([10]))            #初始化偏置项b
y_predict = tf.nn.softmax(tf.matmul(x,W) + b)     #加权变换并进行softmax回归，得到预测概率
print("y_predict:", type(y_predict))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(y_predict),reduction_indices=1))   #求交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)   #用梯度下降法使得残差最小

correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))   #在测试阶段，测试准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                #多个批次的准确度均值

print(type(mnist.train.next_batch(1)))
print(len(mnist.train.next_batch(1)))
print(len(mnist.train.next_batch(1)[0][0]))
print(len(mnist.train.next_batch(1)[1][0]))
batch_xs, batch_ys = mnist.train.next_batch(1)
print(len(batch_ys))
print(len(batch_ys))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):               #训练阶段，迭代1000次
        batch_xs, batch_ys = mnist.train.next_batch(100)           #按批次训练，每批100行数据
        sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})   #执行训练
        if(i%100==0):                  #每训练100次，测试一次
            print("accuracy:",sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))


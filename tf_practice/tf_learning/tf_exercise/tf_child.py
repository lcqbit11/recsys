#!/user/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import csv
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("./MNIST_data1", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y_actual = tf.placeholder(tf.float32, shape=[None, 10])
weights = {
    "W1": tf.Variable(tf.zeros([784,392])),
    "W2": tf.Variable(tf.zeros([392,10]))
}
biases = {
    "b1": tf.Variable(tf.zeros([392])),
    "b2": tf.Variable(tf.zeros([10]))
}

def libsvm2csv(input_libsvm_file, output_csv_file, num_feat):
    reader = csv.reader(open(input_libsvm_file), delimiter=" ")
    writer = csv.writer(open(output_csv_file, 'wb'))
    for line in reader:
        label = line.pop(0)
        if line[-1].strip() == '':
            line.pop(-1)
        line = map(lambda x: tuple(x.split(":")), line)
        new_line = [label] + [0] * num_feat
        for i, v in line:
            i = int(i)
            if i <= num_feat:
                new_line[i] = v
        writer.writerow(new_line)

# def data_transform(raw_data, output_data):



feed_forward1 = tf.contrib.layers.fully_connected(x, 392, tf.nn.relu)
y_predict = tf.nn.softmax(tf.matmul(feed_forward1, weights["W2"]) + biases["b2"])


batch_xs, batch_ys = mnist.train.next_batch(10)
print("batch_xs:", len(batch_xs[0]))
print("batch_xs:", len(batch_xs[1]))
print("batch_xs:", type(batch_xs))
print("batch_xs:", np.shape(batch_xs))



cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(y_predict),reduction_indices=1))   #求交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)   #用梯度下降法使得残差最小

correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))   #在测试阶段，测试准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                #多个批次的准确度均值

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):               #训练阶段，迭代1000次
        batch_xs, batch_ys = mnist.train.next_batch(100)           #按批次训练，每批100行数据
        sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})   #执行训练
        if(i%100==0):                  #每训练100次，测试一次
            print("accuracy:",sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

print(list([1, 2, 3]))
print(tuple([1, 2, 3]))
print(np.array([1, 2, 3]))

a = np.random.random([4, 3])
b1 = np.array([[0, 2], [2, 1]])
b2 = np.array([2, 0])
# b4 = [['2', '4'], ['2', '4']]
# b44 = np.array(list(map(int, b4)))
# print(b44)
b3 = np.array([['0|1|3'], ['0|1|-1']])
b33 = np.array([[int(j) for j in i[0].split('|')] for i in b3])

print(b33)

c1 = tf.nn.embedding_lookup(a, b1)
c2 = tf.nn.embedding_lookup(a, b2)
c3 = tf.nn.embedding_lookup(a, b33)

sess = tf.Session()
print(a)
print(sess.run(c1))
print(sess.run(c2))
print(sess.run(c3))


print(np.mean(sess.run(c1), 1))
print(np.mean(sess.run(c2), 0))
print(np.mean(sess.run(c3), 1))


res = np.array([list([1, 2, 3]), list([2, 3, 4])])
print(res)
print(type(res))
print(res[0])
print(type(res[0]))

print(np.array([1, 2, 3]))
print(list(np.array(list([1, 2, 3]))))






#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

csv = [
  "1,harden|james|curry",
  "2,wrestbrook|harden|durant",
  "3,|paul|towns",
]
TAG_SET = ["harden", "james", "curry", "durant", "paul","towns","wrestbrook"]

ids, post_tags_str = tf.decode_csv(csv, [[-1], [""]])

table = tf.contrib.lookup.index_table_from_tensor(
      mapping=TAG_SET, default_value=-1) ## 这里构造了个查找表 ##
split_tags = tf.string_split(post_tags_str, '|')
# print(split_tags.dense_shape)
tags = tf.SparseTensorValue(
      indices=split_tags.indices,
      values=table.lookup(split_tags.values), ## 这里给出了不同值通过表查到的index ##
      dense_shape=split_tags.dense_shape)
# print(tf.sparse_tensor_to_dense(tags))

TAG_EMBEDDING_DIM = 3
embedding_params = tf.Variable(tf.truncated_normal([len(TAG_SET), TAG_EMBEDDING_DIM]))
embedded_tags = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=tags, sp_weights=None)
tf.nn.embedding_lookup()
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    print(sess.run(embedding_params))
    print(sess.run([embedded_tags]))
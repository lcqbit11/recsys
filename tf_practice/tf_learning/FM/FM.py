from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
"""
使用用户给电影评分数据作为训练样本，其相应的特征为：所有用户id+所有item id。
用户id共943个, item id共1680个，因此特征数量共943+1680=2623个。
"""


def vectorize_dic(dic, ix=None, p=None, n=0, g=0):
    """
    dic -- dictionary of feature lists. Keys are the name of features，{'users':user_array, 'items':item_array}
    ix -- index generator (default None) 索引生成器
    p -- dimension of feature space (number of columns in the sparse matrix) (default None)
    n -- number of sample
    """
    if not ix:
        ix = dict()
    nz = n * g  # n表示样本数量，g表示
    col_ix = np.empty(nz, dtype=int)
    i = 0
    for name, user_item_list in dic.items():
        print("name, user_item_list")
        print(name, user_item_list)
        # lis里面包含了所有的user和item，其数量分别等于样本的数量
        for t in range(len(user_item_list)):
            # 形如ix['943users']，其中943表示用户的id=943，表示用户943出现的次数，即其评价的item的数量，
            # 或者ix['1188items']，其中1188表示itemid=1188，表示item1188出现的次数，即该item被用户评价的次数。
            ix[str(user_item_list[t]) + str(name)] = ix.get(str(user_item_list[t]) + str(name), 0) + 1
            # 偶数位表示每个user出现的次数，奇数位表示每个item出现的次数，长度为g*n，n表示样本的数量
            col_ix[i + t*g] = ix[str(user_item_list[t]) + str(name)]
        i += 1
    row_ix = np.repeat(np.arange(0, n), g)  # [0,1,2]->[0,0,0,,1,1,1,,2,2,2,,]，长度为g*n，n表示样本的数量
    data = np.ones(nz)  # 长度为g*n，n表示样本的数量
    if not p:
        p = len(ix)  # count(distinct user) + count(distinct item)
    print('len ix:', len(ix))
    ixx = np.where(col_ix < p)
    return csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])), shape=(n, p)), ix


def get_batch(data, label=None, batch_size=-1):  # 获取每个batch的样本
    n_samples = data.shape[0]
    if batch_size == -1:
        batch_size = n_samples
    elif batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = data[i:upper_bound]
        if label is not None:
            ret_y = label[i:i + batch_size]
            yield (ret_x, ret_y)


cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('data/ua.base', delimiter='\t', names=cols)  # user=943, item=1680
test = pd.read_csv('data/ua.test', delimiter='\t', names=cols)  # user=943, item=1129


x_train, ix = vectorize_dic({'users': train['user'].values,
                            'items': train['item'].values}, n=len(train.index), g=2)
sparse_feature_name = np.array(list(ix.keys()))
x_test, ix1 = vectorize_dic({'users': test['user'].values,
                           'items': test['item'].values}, ix, x_train.shape[1], n=len(test.index), g=2)
x_train = x_train.todense()
x_test = x_test.todense()
y_train = train['rating'].values
y_test = test['rating'].values

n, feature_num = x_train.shape  # n表示样本数量，p表示特征数量
print('feature_num:', feature_num)
embed_size = 10  # 离散特征对应的向量维度
x = tf.placeholder('float', [None, feature_num])
y = tf.placeholder('float', [None, 1])
w0 = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.zeros([feature_num]))
v = tf.Variable(tf.random_normal([embed_size, feature_num], mean=0, stddev=0.01))  # sparse feature的embedding向量，每个特征embedding成k维，共p个特征
lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')

linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(w, x), 1, keep_dims=True))  # n * 1
cross_terms = 0.5 * tf.reduce_sum(
    tf.subtract(
        tf.pow(tf.matmul(x, tf.transpose(v)), 2),
        tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(v, 2)))
    ), axis=1, keep_dims=True)
y_hat = tf.add(linear_terms, cross_terms)
error = tf.reduce_mean(tf.square(y - y_hat))
l2_norm = tf.reduce_sum(tf.add(tf.multiply(lambda_w, tf.pow(w, 2)), tf.multiply(lambda_v, tf.pow(v, 2))))
loss = tf.add(error, l2_norm)
train_op = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(loss)

epochs = 10
batch_size = 1000
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in tqdm(range(epochs), unit='epoch'):
        perm = np.random.permutation(x_train.shape[0])  # 样本顺序随机打乱
        for bX, bY in get_batch(x_train[perm], y_train[perm], batch_size):
            _, t = sess.run([train_op, loss], feed_dict={x: bX.reshape(-1, feature_num), y: bY.reshape(-1, 1)})
            print("Training loss: %.4f" % t)
        # print("Epoch %d Training loss: %.4f" % (epoch, t))

    v_embedding = sess.run(v)
    sparse_feature_name = sparse_feature_name.reshape([-1, 1])
    v_embedding = v_embedding.T
    name_v_embedding = np.array(np.hstack((sparse_feature_name, v_embedding)))
    with open('./sparse_f_embedding.txt', 'w') as f:
        for line in name_v_embedding:
            for i in range(len(line)):
                f.writelines(line[i] + '\t')
            f.writelines('\n')

    errors = []
    for bX, bY in get_batch(x_test, y_test):
        errors.append(sess.run(error, feed_dict={x: bX.reshape(-1, feature_num), y: bY.reshape(-1, 1)}))
        print("errors:", errors)
    RMSE = np.sqrt(np.array(errors).mean())
    print("RMSE：", RMSE)
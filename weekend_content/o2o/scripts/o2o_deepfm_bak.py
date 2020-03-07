#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from collections import OrderedDict, namedtuple
from itertools import chain
from tensorflow.python.keras.initializers import RandomNormal, Zeros, glorot_normal
from tensorflow.python.keras.layers import Concatenate, Dense, Embedding, Input, Reshape, add, Layer
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import backend as K

def create_singlefeat_inputdict(feature_dim_dict, prefix=''):
    sparse_input = OrderedDict()
    for feat in feature_dim_dict["sparse"]:
        sparse_input[feat.name] = Input(shape=(1,), name=prefix + feat.name, dtype=feat.dtype)
    dense_input = OrderedDict()
    for feat in feature_dim_dict["dense"]:
        dense_input[feat.name] = Input(shape=(1,), name=prefix + feat.name, dtype=feat.dtype)
    return sparse_input, dense_input

def create_embedding_dict(feature_dim_dict, embedding_size, init_std, seed, l2_reg, prefix='sparse_'):
    sparse_embedding = {feat.name: Embedding(feat.dimension, embedding_size,
                                             embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
                                             embeddings_regularizer=l2(l2_reg),
                                             name=prefix + 'emb_' + feat.name) for feat in feature_dim_dict["sparse"]}
    return sparse_embedding

def merge_dense_input(dense_input_, embed_list, embedding_size, l2_reg):
    dense_input = list(dense_input_.values())
    if len(dense_input) > 0:
        continuous_embedding_list = list(map(Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg), ), dense_input))
        continuous_embedding_list = list(map(Reshape((1, embedding_size)), continuous_embedding_list))
        embed_list += continuous_embedding_list
    return embed_list

def get_embedding_vec_list(embedding_dict, input_dict, sparse_fg_list, return_feat_list=()):
    embedding_vec_list = []
    for fg in sparse_fg_list:
        feat_name = fg.name
        if len(return_feat_list) == 0 or feat_name in return_feat_list:
            lookup_idx = input_dict[feat_name]
            embedding_vec_list.append(embedding_dict[feat_name](lookup_idx))
    return embedding_vec_list

def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))

def get_inputs_embedding(feature_dim_dict, embedding_size, l2_reg_embedding, l2_reg_linear, init_std, seed,
                         sparse_input_dict, dense_input_dict, include_linear, prefix=""):
    deep_sparse_emb_dict = create_embedding_dict(feature_dim_dict, embedding_size, init_std, seed, l2_reg_embedding, prefix=prefix + 'sparse')
    deep_emb_list = get_embedding_vec_list(deep_sparse_emb_dict, sparse_input_dict, feature_dim_dict['sparse'])
    deep_emb_list = merge_dense_input(dense_input_dict, deep_emb_list, embedding_size, l2_reg_embedding)
    if include_linear:
        linear_sparse_emb_dict = create_embedding_dict(feature_dim_dict, 1, init_std, seed, l2_reg_linear, prefix + 'linear')
        linear_emb_list = get_embedding_vec_list(linear_sparse_emb_dict, sparse_input_dict, feature_dim_dict['sparse'])
    else:
        linear_emb_list = None

    inputs_list = get_inputs_list([sparse_input_dict, dense_input_dict])
    return inputs_list, deep_emb_list, linear_emb_list

def preprocess_input_embedding(feature_dim_dict, embedding_size, l2_reg_embedding, l2_reg_linear, init_std, seed, create_linear_weight=True):
    sparse_input_dict, dense_input_dict = create_singlefeat_inputdict(feature_dim_dict)
    inputs_list, deep_emb_list, linear_emb_list = get_inputs_embedding(feature_dim_dict, embedding_size,
                                                                       l2_reg_embedding, l2_reg_linear, init_std, seed,
                                                                       sparse_input_dict, dense_input_dict, create_linear_weight)

    return deep_emb_list, linear_emb_list, dense_input_dict, inputs_list

def get_linear_logit(linear_emb_list, dense_input_dict, l2_reg):
    if len(linear_emb_list) > 1:
        linear_term = add(linear_emb_list)
    elif len(linear_emb_list) == 1:
        linear_term = linear_emb_list[0]
    else:
        linear_term = None

    dense_input = list(dense_input_dict.values())
    if len(dense_input) > 0:
        dense_input__ = dense_input[0] if len(dense_input) == 1 else Concatenate()(dense_input)
        linear_dense_logit = Dense(1, activation=None, use_bias=False, kernel_regularizer=l2(l2_reg))(dense_input__)
        if linear_term is not None:
            linear_term = add([linear_dense_logit, linear_term])
        else:
            linear_term = linear_dense_logit

    return linear_term

def activation_layer(activation):
    return tf.keras.layers.Activation(activation)

class FM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):

        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d, expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))
        concated_embeds_value = inputs

        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = tf.reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return None, 1

class DNN(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate,seed=self.seed+i) for i in range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            fc = self.activation_layers[i](fc)
            fc = self.dropout_layers[i](fc,training = training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        # noinspection PyTypeChecker
        return dict(list(base_config.items()) + list(config.items()))

class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss

         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "regression"]:
            raise ValueError("task must be binary or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return None, 1

    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        # noinspection PyTypeChecker
        return dict(list(base_config.items()) + list(config.items()))

class SingleFeat(namedtuple('SingleFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension, dtype="float32"):
        return super(SingleFeat, cls).__new__(cls, name, dimension, dtype)
def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)

def DeepFM(feature_dim_dict, embedding_size=8,
           use_fm=True, dnn_hidden_units=(128, 128), l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0,
           init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the DeepFM Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':['field_1':4,'field_2':3,'field_3':2],'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    deep_emb_list, linear_emb_list, dense_input_dict, inputs_list = preprocess_input_embedding(feature_dim_dict,
                                                                                               embedding_size,
                                                                                               l2_reg_embedding,
                                                                                               l2_reg_linear, init_std,
                                                                                               seed,
                                                                                               create_linear_weight=True)

    linear_logit = get_linear_logit(linear_emb_list, dense_input_dict, l2_reg_linear)

    fm_input = concat_fun(deep_emb_list, axis=1)
    deep_input = tf.keras.layers.Flatten()(fm_input)
    fm_out = FM()(fm_input)
    deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed)(deep_input)
    deep_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(deep_out)

    if len(dnn_hidden_units) == 0 and use_fm == False:  # only linear
        final_logit = linear_logit
    elif len(dnn_hidden_units) == 0 and use_fm == True:  # linear + FM
        final_logit = tf.keras.layers.add([linear_logit, fm_out])
    elif len(dnn_hidden_units) > 0 and use_fm == False:  # linear +　Deep
        final_logit = tf.keras.layers.add([linear_logit, deep_logit])
    elif len(dnn_hidden_units) > 0 and use_fm == True:  # linear + FM + Deep
        final_logit = tf.keras.layers.add([linear_logit, fm_out, deep_logit])
    else:
        raise NotImplementedError

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import date
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import gc
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data_offline = pd.read_csv('../data/ccf_offline_stage1_train.csv')
data_online = pd.read_csv('../data/ccf_online_stage1_train.csv')
test = pd.read_csv('../data/ccf_offline_stage1_test_revised.csv')


data_offline.fillna('null', inplace=True)
data_online.fillna('null', inplace=True)
test = test.fillna('null')
data_offline['Date_received'] = data_offline['Date_received'].astype(str)
data_online['Date_received'] = data_online['Date_received'].astype(str)
data_offline['Date'] = data_offline['Date'].astype(str)
test['Date_received'] = test['Date_received'].astype(str)


# convert Discount_rate and Distance

def getDiscountType(row):
    if row == 'null':
        return 'null'
    elif ':' in row:
        return 1
    else:
        return 0

def convertRate(row):
    """Convert discount to rate"""
    if row == 'null':
        return 1.0
    elif ':' in row:
        rows = row.split(':')
        return 1.0 - float(rows[1])/float(rows[0])
    else:
        return float(row)

def getDiscountMan(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0

def getDiscountJian(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0

def processData(df):

    # convert discunt_rate
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    # convert distance
    df['distance'] = df['Distance'].replace('null', -1).astype(int)
    return df

data_offline = processData(data_offline)
test = processData(test)

def getWeekday(row):
    if row == 'null':
        return row
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1

def getDay(row):
    if row == 'null':
        return row
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).day

data_offline['weekday'] = data_offline['Date_received'].astype(str).apply(getWeekday)
test['weekday'] = test['Date_received'].astype(str).apply(getWeekday)
data_offline['day'] = data_offline['Date_received'].astype(str).apply(getDay)
test['day'] = test['Date_received'].astype(str).apply(getDay)

# weekday_type :  周六和周日为1，其他为0
data_offline['weekday_type'] = data_offline['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
test['weekday_type'] = test['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
def label(row):
    if row['Date_received'] == 'null':
        return -1
    if row['Date'] != 'null':
        td = pd.to_datetime(row['Date'], format='%Y%m%d') -  pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0
data_offline['label'] = data_offline.apply(label, axis=1)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.barplot(x=data_offline[data_offline.label != -1]['label'].value_counts().index,y=data_offline[data_offline.label != -1]['label'].value_counts().values)

feature = data_offline[(data_offline['Date'] < '20160516') | ((data_offline['Date'] == 'null') & (data_offline['Date_received'] < '20160516'))].copy()
dataset = data_offline[(data_offline['Date_received'] >= '20160516') & (data_offline['Date_received'] <= '20160615')].copy()

# 用户聚合特征
fdf = feature.copy()
# key of user
u = fdf[['User_id']].copy().drop_duplicates()
# u_coupon_count : num of coupon received by user
u1 = fdf[fdf['Date_received'] != 'null'][['User_id']].copy()
u1['u_coupon_count'] = 1
u1 = u1.groupby(['User_id'], as_index = False).count()
# u_buy_count : times of user buy offline (with or without coupon)
u2 = fdf[fdf['Date'] != 'null'][['User_id']].copy()
u2['u_buy_count'] = 1
u2 = u2.groupby(['User_id'], as_index = False).count()
# u_buy_with_coupon : times of user buy offline (with coupon)
u3 = fdf[((fdf['Date'] != 'null') & (fdf['Date_received'] != 'null'))][['User_id']].copy()
u3['u_buy_with_coupon'] = 1
u3 = u3.groupby(['User_id'], as_index = False).count()
# u_merchant_count : num of merchant user bought from
u4 = fdf[fdf['Date'] != 'null'][['User_id', 'Merchant_id']].copy()
u4.drop_duplicates(inplace = True)
u4 = u4.groupby(['User_id'], as_index = False).count()
u4.rename(columns = {'Merchant_id':'u_merchant_count'}, inplace = True)
# u_min_distance
utmp = fdf[(fdf['Date'] != 'null') & (fdf['Date_received'] != 'null')][['User_id', 'distance']].copy()
utmp.replace(-1, np.nan, inplace = True)
u5 = utmp.groupby(['User_id'], as_index = False).min()
u5.rename(columns = {'distance':'u_min_distance'}, inplace = True)
u6 = utmp.groupby(['User_id'], as_index = False).max()
u6.rename(columns = {'distance':'u_max_distance'}, inplace = True)
u7 = utmp.groupby(['User_id'], as_index = False).mean()
u7.rename(columns = {'distance':'u_mean_distance'}, inplace = True)
u8 = utmp.groupby(['User_id'], as_index = False).median()
u8.rename(columns = {'distance':'u_median_distance'}, inplace = True)
# merge all the features on key User_id
user_feature = pd.merge(u, u1, on = 'User_id', how = 'left')
user_feature = pd.merge(user_feature, u2, on = 'User_id', how = 'left')
user_feature = pd.merge(user_feature, u3, on = 'User_id', how = 'left')
user_feature = pd.merge(user_feature, u4, on = 'User_id', how = 'left')
user_feature = pd.merge(user_feature, u5, on = 'User_id', how = 'left')
user_feature = pd.merge(user_feature, u6, on = 'User_id', how = 'left')
user_feature = pd.merge(user_feature, u7, on = 'User_id', how = 'left')
user_feature = pd.merge(user_feature, u8, on = 'User_id', how = 'left')
# calculate rate
user_feature['u_use_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float')/user_feature['u_coupon_count'].astype('float')
user_feature['u_buy_with_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float')/user_feature['u_buy_count'].astype('float')
user_feature = user_feature.fillna(0)
# add user feature to dataset on key User_id
dataset2 = pd.merge(dataset, user_feature, on = 'User_id', how = 'left').fillna(0)

del u, u1, u2, u3, u4, u5, u6, u7, u8, utmp, user_feature
gc.collect()

# 商户聚合特征
# key of merchant
m = fdf[['Merchant_id']].copy().drop_duplicates()
# m_coupon_count : num of coupon from merchant
m1 = fdf[fdf['Date_received'] != 'null'][['Merchant_id']].copy()
m1['m_coupon_count'] = 1
m1 = m1.groupby(['Merchant_id'], as_index = False).count()
# m_sale_count : num of sale from merchant (with or without coupon)
m2 = fdf[fdf['Date'] != 'null'][['Merchant_id']].copy()
m2['m_sale_count'] = 1
m2 = m2.groupby(['Merchant_id'], as_index = False).count()
# m_sale_with_coupon : num of sale from merchant with coupon usage
m3 = fdf[(fdf['Date'] != 'null') & (fdf['Date_received'] != 'null')][['Merchant_id']].copy()
m3['m_sale_with_coupon'] = 1
m3 = m3.groupby(['Merchant_id'], as_index = False).count()
# m_min_distance
mtmp = fdf[(fdf['Date'] != 'null') & (fdf['Date_received'] != 'null')][['Merchant_id', 'distance']].copy()
mtmp.replace(-1, np.nan, inplace = True)
m4 = mtmp.groupby(['Merchant_id'], as_index = False).min()
m4.rename(columns = {'distance':'m_min_distance'}, inplace = True)
m5 = mtmp.groupby(['Merchant_id'], as_index = False).max()
m5.rename(columns = {'distance':'m_max_distance'}, inplace = True)
m6 = mtmp.groupby(['Merchant_id'], as_index = False).mean()
m6.rename(columns = {'distance':'m_mean_distance'}, inplace = True)
m7 = mtmp.groupby(['Merchant_id'], as_index = False).median()
m7.rename(columns = {'distance':'m_median_distance'}, inplace = True)
merchant_feature = pd.merge(m, m1, on = 'Merchant_id', how = 'left')
merchant_feature = pd.merge(merchant_feature, m2, on = 'Merchant_id', how = 'left')
merchant_feature = pd.merge(merchant_feature, m3, on = 'Merchant_id', how = 'left')
merchant_feature = pd.merge(merchant_feature, m4, on = 'Merchant_id', how = 'left')
merchant_feature = pd.merge(merchant_feature, m5, on = 'Merchant_id', how = 'left')
merchant_feature = pd.merge(merchant_feature, m6, on = 'Merchant_id', how = 'left')
merchant_feature = pd.merge(merchant_feature, m7, on = 'Merchant_id', how = 'left')
merchant_feature = merchant_feature.fillna(0)
merchant_feature['m_coupon_use_rate'] = merchant_feature['m_sale_with_coupon'].astype('float')/merchant_feature['m_coupon_count'].astype('float')
merchant_feature['m_sale_with_coupon_rate'] = merchant_feature['m_sale_with_coupon'].astype('float')/merchant_feature['m_sale_count'].astype('float')
merchant_feature = merchant_feature.fillna(0)
# add merchant feature to dataset2
dataset3 = pd.merge(dataset2, merchant_feature, on = 'Merchant_id', how = 'left').fillna(0)

del m, m1, m2, m3, m4, m5, m6, m7, mtmp, merchant_feature
gc.collect()

# 用户&商户聚合特征
# key of user and merchant
um = fdf[['User_id', 'Merchant_id']].copy().drop_duplicates()
um1 = fdf[['User_id', 'Merchant_id']].copy()
um1['um_count'] = 1
um1 = um1.groupby(['User_id', 'Merchant_id'], as_index = False).count()
um2 = fdf[fdf['Date'] != 'null'][['User_id', 'Merchant_id']].copy()
um2['um_buy_count'] = 1
um2 = um2.groupby(['User_id', 'Merchant_id'], as_index = False).count()
um3 = fdf[fdf['Date_received'] != 'null'][['User_id', 'Merchant_id']].copy()
um3['um_coupon_count'] = 1
um3 = um3.groupby(['User_id', 'Merchant_id'], as_index = False).count()
um4 = fdf[(fdf['Date_received'] != 'null') & (fdf['Date'] != 'null')][['User_id', 'Merchant_id']].copy()
um4['um_buy_with_coupon'] = 1
um4 = um4.groupby(['User_id', 'Merchant_id'], as_index = False).count()
# merge all user merchant
user_merchant_feature = pd.merge(um, um1, on = ['User_id','Merchant_id'], how = 'left')
user_merchant_feature = pd.merge(user_merchant_feature, um2, on = ['User_id','Merchant_id'], how = 'left')
user_merchant_feature = pd.merge(user_merchant_feature, um3, on = ['User_id','Merchant_id'], how = 'left')
user_merchant_feature = pd.merge(user_merchant_feature, um4, on = ['User_id','Merchant_id'], how = 'left')
user_merchant_feature = user_merchant_feature.fillna(0)
user_merchant_feature['um_buy_rate'] = user_merchant_feature['um_buy_count'].astype('float')/user_merchant_feature['um_count'].astype('float')
user_merchant_feature['um_coupon_use_rate'] = user_merchant_feature['um_buy_with_coupon'].astype('float')/user_merchant_feature['um_coupon_count'].astype('float')
user_merchant_feature['um_buy_with_coupon_rate'] = user_merchant_feature['um_buy_with_coupon'].astype('float')/user_merchant_feature['um_buy_count'].astype('float')
user_merchant_feature = user_merchant_feature.fillna(0)
# add user_merchant_feature to data3
dataset4 = pd.merge(dataset3, user_merchant_feature, on = ['User_id','Merchant_id'], how = 'left').fillna(0)

del um, um1, um2, um3, um4, user_merchant_feature
gc.collect()


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

dataset4 = reduce_mem_usage(dataset4)

sparse_features = ['User_id', 'Merchant_id', 'Coupon_id', 'weekday', 'discount_type', 'weekday_type']
dense_features = [fea for fea in dataset4.columns if fea not in sparse_features and fea not in ['Discount_rate', 'Distance', 'Date_received', 'Date', 'label']]

dataset4[sparse_features] = dataset4[sparse_features].fillna('null', )
dataset4[dense_features] = dataset4[dense_features].fillna(0, )



log_features = ['u_coupon_count', 'u_buy_count','u_buy_with_coupon', 'u_merchant_count', 'u_min_distance',
       'u_max_distance', 'u_mean_distance', 'u_median_distance',
       'u_use_coupon_rate', 'u_buy_with_coupon_rate', 'm_coupon_count',
       'm_sale_count', 'm_sale_with_coupon', 'm_min_distance',
       'm_max_distance', 'm_mean_distance', 'm_median_distance',
       'm_coupon_use_rate', 'm_sale_with_coupon_rate', 'um_count',
       'um_buy_count', 'um_coupon_count', 'um_buy_with_coupon', 'um_buy_rate',
       'um_coupon_use_rate', 'um_buy_with_coupon_rate']

for fea in tqdm(log_features):
    dataset4[fea] = dataset4[fea].apply(lambda x: np.log1p(x))

for feat in tqdm(sparse_features):
    lbe = LabelEncoder()
    dataset4[feat] = lbe.fit_transform(dataset4[feat])
mms = MinMaxScaler(feature_range=(0, 1))
dataset4[dense_features] = mms.fit_transform(dataset4[dense_features])

sparse_feature_list = [SingleFeat(feat, dataset4[feat].nunique()) for feat in sparse_features]
dense_feature_list = [SingleFeat(feat, 0,) for feat in dense_features]
train, valid = train_test_split(dataset4, test_size=0.2, stratify=dataset4['label'], random_state=2019)
train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
                    [train[feat.name].values for feat in dense_feature_list]
valid_model_input = [valid[feat.name].values for feat in sparse_feature_list] + \
                   [valid[feat.name].values for feat in dense_feature_list]
checkpoint_predictions = []
weights = []

for model_idx in range(2):
    print('【', 'model_{}'.format(model_idx + 1), '】')
    model = DeepFM(
        {"sparse": sparse_feature_list, "dense": dense_feature_list},
        dnn_hidden_units=(64, 64),
        dnn_use_bn=True,
        task='binary')
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
    for global_epoch in range(2):
        print('【', 'global_epoch_{}'.format(global_epoch + 1), '】')
        model.fit(
            train_model_input,
            train['label'].values,
            batch_size=64,
            epochs=1,
            verbose=1)
        checkpoint_predictions.append(model.predict(valid_model_input, batch_size=64).flatten())
        weights.append(2 ** global_epoch)


predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
# avgAUC calculation
valid1 = valid.copy()
valid1['pred_prob'] = list(predictions)
vg = valid1.groupby(['Coupon_id'])
aucs = []
for i in vg:
    tmpdf = i[1]
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print('auc: ', np.average(aucs))
# -*- coding : utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

from tensorflow.contrib import slim
from models.nets.resnet_v2 import resnet_v2
import os
import time
import sys
import numpy as np
from time import time

M = 61
k = 10
def teacher(input_images, 
            keep_prob,
            is_training=True,
            weight_decay=5e-5,
            batch_norm_decay=0.99,
            batch_norm_epsilon=0.001):
    with tf.variable_scope("Teacher_model"):     
        net, endpoints = resnet_v2(inputs=input_images,
                                num_classes=M,
                                is_training=True,
                                scope='resnet_v2')
        # co_trained layers
        var_scope = 'Teacher_model/resnet_v2/'
        co_list_0 = slim.get_model_variables(var_scope + 'Conv2d_0')
        # co_list_1 = slim.get_model_variables(var_scope +'InvertedResidual_16_0/conv')
        # co_list_2 = slim.get_model_variables(var_scope +'InvertedResidual_24_')
        t_co_list = co_list_0
        
        base_var_list = slim.get_model_variables('Teacher_model/resnet_v2')

        # feature & attention
        t_g0 = endpoints["InvertedResidual_{}_{}".format(256, 2)]
        t_at0 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(t_g0), -1), axis=0, name='t_at0')
        t_g1 = endpoints["InvertedResidual_{}_{}".format(512, 3)]
        t_at1 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(t_g1), -1), axis=0, name='t_at1')
        part_feature = endpoints["InvertedResidual_{}_{}".format(1024, 3)]
        t_at2 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(part_feature), -1), axis=0, name='t_at2')
        t_g3 = endpoints["InvertedResidual_{}_{}".format(1024, 4)]
        t_at3 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(t_g3), -1), axis=0, name='t_at3')
        object_feature = endpoints["InvertedResidual_{}_{}".format(1024, 5)]
        t_at4 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(object_feature), -1), axis=0, name='t_at4')
        
        t_g = (t_g0, t_g1, part_feature, object_feature)
        t_at = (t_at0, t_at1, t_at2, t_at3, t_at4)

        object_feature_h = object_feature.get_shape().as_list()[1]
        object_feature_w = object_feature.get_shape().as_list()[2]
        fc_obj = slim.max_pool2d(object_feature, (object_feature_h, object_feature_w), scope="GMP1")
        batch_norm_params = {
            'center': True,
            'scale': True,
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
        }

        fc_obj = slim.conv2d(fc_obj,
                            M,
                            [1, 1],
                            activation_fn=None,    
                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            scope='fc_obj')
        fc_obj = tf.nn.dropout(fc_obj, keep_prob=keep_prob)
        fc_obj = slim.flatten(fc_obj)
        fc_part = slim.conv2d(part_feature,
                            M * k,          #卷积核个数
                            [1, 1],         #卷积核高宽
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,                               # 标准化器设置为BN
                            normalizer_params=batch_norm_params,
                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
                            )
        fc_part_h = fc_part.get_shape().as_list()[1]
        fc_part_w = fc_part.get_shape().as_list()[2]
        fc_part = slim.max_pool2d(fc_part, (fc_part_h, fc_part_w), scope="GMP2")
        ft_list = tf.split(fc_part,
                        num_or_size_splits=M,
                        axis=-1)            #最后一维度（C）
        cls_list = []
        for i in range(M):
            ft = tf.transpose(ft_list[i], [0, 1, 3, 2])
            cls = layers_lib.pool(ft,
                                [1, k],
                                "AVG")
            cls = layers.flatten(cls)
            cls_list.append(cls)
        fc_ccp = tf.concat(cls_list, axis=-1) #cross_channel_pooling (N, M)

        fc_part = slim.conv2d(fc_part,
                            M,
                            [1, 1],
                            activation_fn=None,                         
                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            scope="fc_part")
        fc_part = tf.nn.dropout(fc_part, keep_prob=keep_prob)
        fc_part = slim.flatten(fc_part)
        t_var_list = slim.get_model_variables()
    return t_co_list, t_g, t_at, fc_obj, fc_part, fc_ccp, base_var_list, t_var_list
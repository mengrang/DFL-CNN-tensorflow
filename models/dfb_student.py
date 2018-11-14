#coding : utf-8
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
from nets import studentnet_v2
from resnet import resnet_v2
import os
import h5py
import time
import sys
import numpy as np
from time import time

M = 61
k = 10
def student(input_images, 
            keep_prob,
            is_training=True,
            weight_decay=0.00004,
            batch_norm_decay=0.99,
            batch_norm_epsilon=0.001):
    with tf.variable_scope("Student_model"):
        net, endpoints = studentnet_v2.studentnet_v2(inputs=input_images,
                                num_classes=61,
                                is_training=True,
                                scope='student')
        # co_trained layers
        var_scope = 'Student_model/student/'
        co_var_list_0 = slim.get_model_variables(var_scope + 'Conv2d_0')
        co_var_list_1 = slim.get_model_variables(var_scope +'InvertedResidual_16_0/conv')
        # co_var_list_2 = slim.get_model_variables(var_scope +'InvertedResidual_24_')
        s_co_list = co_var_list_0 + co_var_list_1
        # feature & attention
        s_g32 = endpoints["InvertedResidual_{}_{}".format(32, 1)]
        s_at32 = tf.reduce_sum(tf.square(s_g32), -1, name='s_at32')
        s_g0 = endpoints["InvertedResidual_{}_{}".format(64, 1)]
        s_at0 = tf.reduce_sum(tf.square(s_g0), -1, name='s_at0')
        s_g1 = endpoints["InvertedResidual_{}_{}".format(96, 1)]
        s_at1 = tf.reduce_sum(tf.square(s_g1), -1, name='s_at1')
        s_g2 = endpoints["InvertedResidual_{}_{}".format(160, 1)]
        s_at2 = tf.reduce_sum(tf.square(s_g2), -1, name='s_at2')
        s_g3 = endpoints["InvertedResidual_{}_{}".format(256, 0)]
        s_at3 = tf.reduce_sum(tf.square(s_g3), -1, name='s_at4')
        s_g4 = endpoints["Conv2d_8"]
        s_at4 = tf.reduce_sum(tf.square(s_g4), -1, name='s_at5')
        
        s_g = (s_g32, s_g0, s_g1, s_g2, s_g4)
        s_at = (s_at32, s_at0, s_at1, s_at2, s_at3, s_at4)
    
        s_part_feature = s_g2
        s_object_feature = s_g4
        # print(s_object_feature.get_shape())
        # exit()
        base_var_list = slim.get_model_variables('Student_model/student')

        batch_norm_params = {
            'center': True,
            'scale': True,
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
        }   
        # obj 
        s_fc_obj = slim.max_pool2d(s_object_feature, (6, 8), scope="s_GMP1")       
        s_fc_obj = slim.conv2d(s_fc_obj,
                            num_outputs=M,
                            kernel_size=[1, 1],
                            activation_fn=None,                       
                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            scope='s_fc_obj')
        s_fc_obj = tf.nn.dropout(s_fc_obj, keep_prob=keep_prob)
        s_fc_obj = layers.flatten(s_fc_obj)

        
        s_fc_part = slim.conv2d(s_part_feature,
                            M * k,                #卷积核个数
                            [1, 1],               #卷积核高宽
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,                               # 标准化器设置为BN
                            normalizer_params=batch_norm_params,
                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
                            )
        s_fc_part = slim.max_pool2d(s_fc_part, (6, 8), scope="s_GMP2")
        s_ft_list = tf.split(s_fc_part,
                        num_or_size_splits=M,
                        axis=-1)                  #最后一维度（C）
        s_cls_list = []
        for i in range(M):
            s_ft = tf.transpose(s_ft_list[i], [0, 1, 3, 2])
            s_cls = layers_lib.pool(s_ft,
                                [1, k],
                                "AVG")
            s_cls = layers.flatten(s_cls)
            s_cls_list.append(s_cls)
        s_fc_ccp = tf.concat(s_cls_list, axis=-1)    #cross_channel_pooling (N, M)

        s_fc_part = slim.conv2d(s_fc_part,
                            M,
                            [1, 1],
                            activation_fn=None,
                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            scope="s_fc_part")
        s_fc_part = tf.nn.dropout(s_fc_part, keep_prob=keep_prob)
        s_fc_part = slim.flatten(s_fc_part)

        s_fc_list = []
        s_var_list = slim.get_model_variables('Student_model')
        # for var in s_var_list:
        #     if var not in base_var_list_0:
        #         s_fc_list.append(var)
        # print('base_var_list',base_var_list)
        # exit()
    return s_g, s_at, s_fc_obj, s_fc_part, s_fc_ccp, s_co_list, base_var_list, s_var_list
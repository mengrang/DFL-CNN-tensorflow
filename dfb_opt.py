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
import os
import sys
import numpy as np

def train(loss_val, base_var_list, var_list, lr, clip_value):   
    opt = tf.train.AdamOptimizer
    fc_optimizer = opt(learning_rate=lr)
    net_optimizer = opt(learning_rate=lr*0.01)
    grads = tf.gradients(loss_val, var_list)
    net_grads = grads[:len(base_var_list)]
    fc_grads = grads[len(base_var_list):]
    clipped_net_grads = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in zip(net_grads, var_list[:len(base_var_list)]) if grad is not None]
    clipped_fc_grads = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in zip(fc_grads, var_list[len(base_var_list):]) if grad is not None]
    for grad, var in clipped_fc_grads:
        tf.summary.histogram(var.op.name + "/gradient", grad) 
    for grad, var in clipped_net_grads:
        tf.summary.histogram(var.op.name + "/gradient", grad)
    train_fc = fc_optimizer.apply_gradients(clipped_fc_grads)
    train_net = net_optimizer.apply_gradients(clipped_net_grads)
    train_op = tf.group(train_fc, train_net)
    return train_op
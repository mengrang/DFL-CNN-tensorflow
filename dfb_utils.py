#coding : utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim
import os
import sys
import numpy as np

def accuracy_top1(y_true, predictions):
    acc_top1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(predictions, axis=-1)), tf.float32), axis=-1)
    return acc_top1

def accuracy_top5(y_true, predictions):
    acc_top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions, tf.argmax(y_true, axis=-1), k=5), tf.float32), axis=-1)
    return acc_top5   
    
def accuracy_top3(y_true, predictions):
    acc_top3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions, tf.argmax(y_true, axis=-1), k=3), tf.float32), axis=-1)
    return acc_top3        

def focal_loss(targets, logits):   
    one_vector = tf.ones(logits.get_shape().as_list(), logits.dtype.base_dtype)
    _epsilon = tf.convert_to_tensor(1e-12, logits.dtype.base_dtype)
    logits = tf.clip_by_value(logits, _epsilon, 1. - _epsilon)
    return tf.reduce_mean(-tf.reduce_sum((one_vector - logits) ** 2 * targets * tf.log(logits), axis=-1), axis=0)

def smooth_l1_loss(targets, logits):
    one_vector = tf.ones(logits.get_shape().as_list(), logits.dtype.base_dtype)
    smoothl1_loss = 0.5*tf.reduce_mean(tf.cast(tf.less(tf.abs(logits-targets), one_vector),tf.float32)*tf.square(logits-targets)) \
                    + tf.reduce_mean((one_vector-tf.cast(tf.less(tf.abs(logits-targets), one_vector),tf.float32))*(tf.abs(logits-targets)-0.5*one_vector))
    return smoothl1_loss

def ohkm(loss, batch_size):
    ohkm_loss = 0.
    for i in range(batch_size):
        sub_loss = loss[i]
        topk_val, topk_idx = tf.nn.top_k(sub_loss, 
                                        k=8, 
                                        sorted=False, name='ohkm{}'.format(i))
        tmp_loss = tf.gather(sub_loss, topk_idx, name='ohkm_loss{}'.format(i)) # can be ignore ???
        ohkm_loss += tf.reduce_sum(tmp_loss) / 8
    ohkm_loss /= batch_size
    return ohkm_loss    
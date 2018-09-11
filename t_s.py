# -*-coding:utf-8-*-
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
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16
from mobilenet_v1 import mobilenet_v1
from mobilenet_v2 import mobilenet_v2
import studentnet_v2
import os
import h5py
import time
import sys
from tflearn.data_utils import shuffle
import numpy as np
from time import time

"""
" Log Configuration
"""

tf.app.flags.DEFINE_string(name="data_dir", default="./datasets", help="The directory to the dataset.")

tf.app.flags.DEFINE_string(name="t_s_logs_dir", default="./logs/", help="The directory to the model checkpoint, tensorboard and log.")

tf.app.flags.DEFINE_string(name="teacher_dir", default="./logs/v2/teacher", help="The directory to the pre-trained  teachernet weights.")

tf.app.flags.DEFINE_integer(name="batch_size", default=128, help="The number of samples in each batch.")

tf.app.flags.DEFINE_integer(name="num_class", default=12, help="The number of classes.")

tf.app.flags.DEFINE_integer(name="epoches", default=300, help="The number of training epoch.") 

tf.app.flags.DEFINE_integer(name="verbose", default=4, help="The number of training step to show the loss and accuracy.")

tf.app.flags.DEFINE_integer(name="patience", default=1, help="The patience of the early stop.")

tf.app.flags.DEFINE_boolean(name="debug", default=True, help="Debug mode or not")

tf.app.flags.DEFINE_float(name="alpha",default=0.0, help="The weight of knowledge distillation ")

tf.app.flags.DEFINE_float(name="beta", default=300.0, help="The weight of knowledge distillation ")

tf.app.flags.DEFINE_float(name="T", default=0.8, help="The temprature of knowledge distillation ")

tf.app.flags.DEFINE_boolean(name="mimic",default=True, help="MImic mode or not")

tf.flags.DEFINE_string(name="mode", default="train", help="Mode train/ test/ visualize")

"""
" Local Parameters
"""
tf.app.flags.DEFINE_integer(name="input_scale",
                            default=224,
                            help="The scale of the input images.")
tf.app.flags.DEFINE_integer(name="num_atn",
                            default=4,
                            help="The number of attention of each image.")
FLAGS = tf.app.flags.FLAGS

M = FLAGS.num_class
k = 10
def teacher(input_images, 
            keep_prob,
            is_training=True,
            weight_decay=0.00004,
            batch_norm_decay=0.997,
            batch_norm_epsilon=0.001):
    with tf.variable_scope("Teacher_model"):     
        net, endpoints = mobilenet_v2(inputs=input_images,
                                num_classes=FLAGS.num_class,
                                is_training=True,
                                spatial_squeeze=True,
                                scope='mobilenet_v2')
        
        base_var_list = slim.get_model_variables()
        for _ in range(2):
             base_var_list.pop()

        # feature & attention
        t_g0 = endpoints["InvertedResidual_{}_{}".format(64, 3)]
        t_at0 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(t_g0), -1), axis=0, name='t_at0')
        t_g1 = endpoints["InvertedResidual_{}_{}".format(96, 2)]
        t_at1 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(t_g1), -1), axis=0, name='t_at1')
        part_feature = endpoints["InvertedResidual_{}_{}".format(160, 2)]
        t_at2 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(part_feature), -1), axis=0, name='t_at3')
        t_p_o = endpoints["InvertedResidual_{}_{}".format(320, 0)]
        t_at3 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(t_p_o), -1), axis=0, name='t_at4')
        object_feature = endpoints["Conv2d_8"]
        t_at4 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(object_feature), -1), axis=0, name='t_at5')
       
        t_g = (t_g0, t_g1, part_feature, object_feature)
        t_at = (t_at0, t_at1, t_at2, t_at3, t_at4)
        
        fc_obj = slim.max_pool2d(object_feature, (6, 8), scope="GMP1")
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
                            M * k,          
                            [1, 1],         
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,                               
                            normalizer_params=batch_norm_params,
                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
                            )

        fc_part = slim.max_pool2d(fc_part, (6, 8), scope="GMP2")
        ft_list = tf.split(fc_part,
                        num_or_size_splits=FLAGS.num_class,
                        axis=-1)            
        cls_list = []
        for i in range(M):
            ft = tf.transpose(ft_list[i], [0, 1, 3, 2])
            cls = layers_lib.pool(ft,
                                [1, 10],
                                "AVG")
            cls = layers.flatten(cls)
            cls_list.append(cls)
        fc_ccp = tf.concat(cls_list, axis=-1) 

        fc_part = slim.conv2d(fc_part,
                            FLAGS.num_class,
                            [1, 1],
                            activation_fn=None,                         
                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            scope="fc_part")
        fc_part = tf.nn.dropout(fc_part, keep_prob=keep_prob)
        fc_part = slim.flatten(fc_part)
        t_var_list = slim.get_model_variables()
    return t_g, t_at, fc_obj, fc_part, fc_ccp, base_var_list, t_var_list

def student(input_images, 
            keep_prob,
            is_training=True,
            weight_decay=0.00004,
            batch_norm_decay=0.997,
            batch_norm_epsilon=0.001):
    with tf.variable_scope("Student_model"):
        net, endpoints = studentnet_v2.studentnet_v2(inputs=input_images,
                                num_classes=FLAGS.num_class,
                                is_training=True,
                                scope='student')
        
        # feature & attention
        s_g0 = endpoints["InvertedResidual_{}_{}".format(64, 1)]
        s_at0 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(s_g0), -1), axis=0, name='s_at0')
        s_g1 = endpoints["InvertedResidual_{}_{}".format(96, 1)]
        s_at1 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(s_g1), -1), axis=0, name='s_at1')
        s_g2 = endpoints["InvertedResidual_{}_{}".format(160, 1)]
        s_at2 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(s_g2), -1), axis=0, name='s_at2')
        s_g3 = endpoints["InvertedResidual_{}_{}".format(256, 0)]
        s_at3 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(s_g3), -1), axis=0, name='s_at4')
        s_g4 = endpoints["Conv2d_8"]
        s_at4 = tf.nn.l2_normalize(tf.reduce_sum(tf.square(s_g4), -1), axis=0, name='s_at5')
        
        s_g = (s_g0, s_g1, s_g2, s_g4)
        s_at = (s_at0, s_at1, s_at2, s_at3, s_at4)
    
        s_part_feature = s_g2
        s_object_feature = s_g4
        
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
                            M * k,                
                            [1, 1],               
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,                               
                            normalizer_params=batch_norm_params,
                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
                            )
        s_fc_part = slim.max_pool2d(s_fc_part, (6, 8), scope="s_GMP2")
        s_ft_list = tf.split(s_fc_part,
                        num_or_size_splits=FLAGS.num_class,
                        axis=-1)                  
        s_cls_list = []
        for i in range(M):
            s_ft = tf.transpose(s_ft_list[i], [0, 1, 3, 2])
            s_cls = layers_lib.pool(s_ft,
                                [1, 10],
                                "AVG")
            s_cls = layers.flatten(s_cls)
            s_cls_list.append(s_cls)
        s_fc_ccp = tf.concat(s_cls_list, axis=-1)    

        s_fc_part = slim.conv2d(s_fc_part,
                            FLAGS.num_class,
                            [1, 1],
                            activation_fn=None,
                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            scope="s_fc_part")
        s_fc_part = tf.nn.dropout(s_fc_part, keep_prob=keep_prob)
        s_fc_part = slim.flatten(s_fc_part)

        s_fc_list = []
        s_var_list = slim.get_model_variables('Student_model')
        for var in s_var_list:
            if var not in base_var_list:
                s_fc_list.append(var)
        # print('base_var_list',base_var_list)
        # exit()
    return s_g, s_at, s_fc_obj, s_fc_part, s_fc_ccp, base_var_list, s_var_list

def train(loss_val, base_var_list, var_list, lr, clip_value): 
    print('base_var_list:{b},var_list:{v}'.format(b=len(base_var_list),v=len(var_list)))


    opt = tf.train.AdamOptimizer
    fc_optimizer = opt(learning_rate=lr)
    net_optimizer = opt(learning_rate=lr * 1e-2)
    grads = tf.gradients(loss_val, var_list)
    net_grads = grads[:len(base_var_list)]
    fc_grads = grads[len(base_var_list):]
    clipped_net_grads = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in zip(net_grads, var_list[:len(base_var_list)]) \
                                if grad is not None]
    clipped_fc_grads = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in zip(fc_grads, var_list[len(base_var_list):]) \
                            if grad is not None]
    if FLAGS.debug:
        if grad is not None:
            for grad, var in clipped_fc_grads:
                tf.summary.histogram(var.op.name + "/gradient", grad) 
            for grad, var in clipped_net_grads:
                tf.summary.histogram(var.op.name + "/gradient", grad)
    train_fc = fc_optimizer.apply_gradients(clipped_fc_grads)
    train_net = net_optimizer.apply_gradients(clipped_net_grads)
    train_op = tf.group(train_fc, train_net)
    
    return train_op

def accuracy_top1(y_true, predictions):
    acc_top1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(predictions, axis=-1)), tf.float32), axis=-1)
    return acc_top1

def accuracy_top5(y_true, predictions):
    acc_top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions, tf.argmax(y_true, axis=-1), k=5), tf.float32), axis=-1)
    return acc_top5         


def main(argv=None):
    is_training = True
    input_images = tf.placeholder(dtype=tf.float32,
                                      shape=[FLAGS.batch_size, 192, 256, 3],
                                      name="input_images")
    y_true = tf.placeholder(dtype=tf.float32,
                            shape=[FLAGS.batch_size, FLAGS.num_class],
                            name="y_true")
    keep_prob = tf.placeholder(dtype=tf.float32,
                                name="dropout")
    learning_rate = tf.placeholder(dtype=tf.float64,
                                    name="learning_rate")
    clip_value = tf.placeholder(dtype=tf.float32,
                                name="clip_value")
        
    """
    ""inference
    """
    t_g, t_at, fc_obj, fc_part, fc_ccp, base_var_list, t_var_list= teacher(input_images, keep_prob, is_training=is_training)
    fc_part = tf.nn.softmax(fc_part)
    fc_ccp = tf.nn.softmax(fc_ccp)
    fc_obj = tf.nn.softmax(fc_obj)
    t_predictions = (fc_part + 0.1 * fc_ccp + fc_obj) / 3.
    
    """
    ""teachernet loss
    """
    if not FLAGS.mimic:
        obj_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=fc_obj))
        part_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=fc_part))
        ccp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=fc_ccp))
        loss = 0.1 * ccp_loss + part_loss + obj_loss
        acc_top1 = accuracy_top1(y_true, t_predictions)
        acc_top5 = accuracy_top5(y_true, t_predictions)
        """
        ""Summary
        """
        tf.summary.scalar("t_obj_loss", obj_loss)
        tf.summary.scalar("t_part_loss", part_loss)
        tf.summary.scalar("t_ccp_loss", ccp_loss)
        tf.summary.scalar("t_loss", loss)
        tf.summary.scalar("acc_top1", acc_top1)
        tf.summary.scalar("acc_top5", acc_top5)

        train_op = train(loss, base_var_list, t_var_list, learning_rate, clip_value)                
        print("Setting up summary op...")
        summary_op = tf.summary.merge_all()

    """
    ""teacher-student loss
    """
    if FLAGS.mimic:
        s_g, s_at, s_fc_obj, s_fc_part, s_fc_ccp, base_var_list, s_var_list= student(input_images, keep_prob, is_training=is_training)
        # predictions
        # print(sess.run(s_var_list))
        # exit()
        predictions = (tf.nn.softmax(s_fc_part) + 0.1*tf.nn.softmax(s_fc_ccp) + tf.nn.softmax(s_fc_obj)) / 3.0

        tau_obi_loss = FLAGS.T * FLAGS.T * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                                                logits=tf.scalar_mul(1.0 / FLAGS.T, s_fc_obj),
                                                                labels=tf.scalar_mul(1.0 / FLAGS.T, fc_obj)                                                     
                                                                                                  ))
        s_obj_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=s_fc_obj))  
        obj_loss = (1 - FLAGS.alpha) * s_obj_loss + FLAGS.alpha * tau_obi_loss
        # part_loss
        tau_part_loss = FLAGS.T * FLAGS.T * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                                                logits=tf.scalar_mul(1.0 / FLAGS.T, s_fc_part),
                                                                labels=tf.scalar_mul(1.0 / FLAGS.T, fc_part)
                                                                
                                                                                                   ))
        s_part_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=s_fc_part))      
        part_loss = (1 - FLAGS.alpha) * s_part_loss + FLAGS.alpha * tau_part_loss
        # ccp_loss
        tau_ccp_loss = FLAGS.T * FLAGS.T * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                                                logits=tf.scalar_mul(1.0 / FLAGS.T, s_fc_ccp),
                                                                labels=tf.scalar_mul(1.0 / FLAGS.T, fc_ccp)                                                             
                                                                                                  ))    
        s_ccp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=s_fc_ccp, labels=y_true))    
        ccp_loss = (1 - FLAGS.alpha) * s_ccp_loss + FLAGS.alpha * tau_ccp_loss
        # attention transfer
        # at_loss
        at_losses = [tf.reduce_mean(tf.square(x - y) + 0.0 * tf.abs(x - y)) for x, y in zip(s_at, t_at)]
        at_loss = sum(at_losses)
        # fm loss
        fm_losses = [tf.reduce_mean(tf.square(x - y) + 0.0 * tf.abs(x - y)) for x, y in zip(s_g, t_g)]
        fm_loss = sum(fm_losses)
        # kd_loss
        kd_loss = FLAGS.T * FLAGS.T * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                                                logits=tf.scalar_mul(1.0 / FLAGS.T, predictions),
                                                                labels=tf.scalar_mul(1.0 / FLAGS.T, t_predictions)))
        # total_loss
        s_loss = 0.1 * s_ccp_loss + s_part_loss + s_obj_loss    
        loss = 2 * ((1 - FLAGS.alpha) * s_loss + FLAGS.alpha * (0.4 * ccp_loss + part_loss + obj_loss)) + FLAGS.beta * at_loss +  0.05*fm_loss + 0.5*kd_loss
        
        # acc_top1
        acc_top1 = accuracy_top1(y_true, predictions)
        # acc_top5
        acc_top5 = accuracy_top5(y_true, predictions)
        """
        ""Summary
        """
        tf.summary.scalar("obj_loss", obj_loss)
        tf.summary.scalar("part_loss", part_loss)
        tf.summary.scalar("ccp_loss", ccp_loss)
        tf.summary.scalar("at_loss", at_loss)
        tf.summary.scalar("fm_loss", fm_loss)
        tf.summary.scalar("kd_loss", kd_loss)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("acc_top1", acc_top1)
        tf.summary.scalar("acc_top5", acc_top5)
        """
        " back propagate
        """
        train_op = train(loss, base_var_list, s_var_list, learning_rate, clip_value)                 
        print("Setting up summary op...")
        summary_op = tf.summary.merge_all()

    """
    " Loading Data
    """
    print("Loading Data......")
    if FLAGS.mode == 'train':
        with h5py.File(os.path.join(FLAGS.data_dir, "trainset.h5"), "r") as f:
            X_train = f["X"][:]
            Y_train = f["Y"][:]
        
            print(X_train.shape)
            print(Y_train.shape)
            f.close()
            print("\tLoaded Train Data......")
    with h5py.File(os.path.join(FLAGS.data_dir, "testset.h5"), "r") as f:
        X_test = f["X"][:]
        Y_test = f["Y"][:]
        
        print(X_test.shape)
        print(Y_test.shape)
        f.close()
        print("\tLoaded Test Data......")  
    print("Verifying the data......")
    # for i in range(100):
    #     if i*200 > 6000:
    #         break
    #     train_img = X_train[i*100]
    #     test_img = X_test[i*200]
    #     cv2.imshow("Train", np.uint8(train_img))
    #     cv2.imshow("Test", np.uint8(test_img))
    #
    #     print(i, "Train class id", np.argmax(Y_train[i*100], axis=0))
    #     print(i, "Test class id", np.argmax(Y_test[i*200], axis=0))
    #     cv2.waitKey()
    # exit()
    
    """
    " Setting up Saver
    """
    print("Setting up Saver...")
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    saver_s = tf.train.Saver(s_var_list,max_to_keep=3)
    saver_t = tf.train.Saver(t_var_list, max_to_keep=3)
    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.t_s_logs_dir, 'train'),
                                        sess.graph)
    
    valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.t_s_logs_dir, 'valid'),
                                        sess.graph)

    print("Initialize global variables")                                   
    sess.run(tf.global_variables_initializer())

    """
    " Resume
    """
    ckpt_t = tf.train.get_checkpoint_state(FLAGS.teacher_dir)
    if ckpt_t and ckpt_t.model_checkpoint_path:
        print('teacher:', ckpt_t.model_checkpoint_path)
        saver_t.restore(sess, ckpt_t.model_checkpoint_path)
        print("Model restored...")

    
    ckpt = tf.train.get_checkpoint_state(FLAGS.t_s_logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('student:', ckpt.model_checkpoint_path)
        saver_s.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    """
    " Training...
    """
    if FLAGS.mode == 'train':      
        train_batch = int(X_train.shape[0] / FLAGS.batch_size)
        valid_batch = int(X_test.shape[0] / FLAGS.batch_size)
        last_loss = 10000.
        patience = 0
        best_acc = 0.0
        clipvalue = 0.0005
        global_step = tf.train.get_or_create_global_step()
        epoch_st = global_step // train_batch + 1
        
        current = 0.0001
        for epoch in range(58, FLAGS.epoches if FLAGS.debug else 1):
        
            print("Epoch %i ----> Starting......" % epoch)
            X_train, Y_train = shuffle(X_train, Y_train)
            start_time = time()
            
            """
            " Build learning rate
            """
            if epoch <= 20:
                lr = 1e-3 / 20.0 * epoch
                current = lr
            elif epoch > 20 and epoch <= 40:
                lr = current
            elif epoch > 40:
                lr = current
            
            for step in range(train_batch):
                batch_x = X_train[step * FLAGS.batch_size: (step + 1) * FLAGS.batch_size]
                batch_y = Y_train[step * FLAGS.batch_size: (step + 1) * FLAGS.batch_size]
                summary, _ = sess.run([summary_op, train_op],
                                    feed_dict={input_images: batch_x,
                                                y_true: batch_y,
                                                keep_prob: 1.,
                                                learning_rate: lr,
                                                clip_value: clipvalue})
                train_writer.add_summary(summary, step + train_batch * (epoch-1))
                """
                ' print the train loss
                """
                if (epoch * train_batch + step) % FLAGS.verbose == 0:
                    if not FLAGS.mimic:
                        loss_t, loss_ct, loss_ot, loss_pt, acc_1t, acc_5t = \
                            sess.run([loss, ccp_loss, obj_loss, part_loss, acc_top1, acc_top5],
                                    feed_dict={input_images: batch_x,
                                                y_true: batch_y,
                                                keep_prob: 0.7,
                                                learning_rate: lr,
                                                clip_value: clipvalue})
                        print("Step %i, Train_loss %0.4f, ccp_loss %0.4f, obj_loss %0.4f, part_loss %0.4f, acc_1 %0.4f, acc_5 %0.4f" %
                                    ((epoch-1) * train_batch + step, loss_t, loss_ct, loss_ot, loss_pt, acc_1t, acc_5t))
                    if FLAGS.mimic:
                        loss_t, loss_ct, loss_ot, loss_pt, loss_at, loss_fm, loss_kd, acc_1t, acc_5t = \
                        sess.run([loss, ccp_loss, obj_loss, part_loss, at_loss, fm_loss, kd_loss, acc_top1, acc_top5],
                                feed_dict={input_images: batch_x,
                                            y_true: batch_y,
                                            keep_prob: 1.,
                                            learning_rate: lr,
                                            clip_value: clipvalue})
                        print("Step: %i, Loss: \33[91m%.4f\033[0m, ccp_loss: %.4f, obj_loss: %.4f, part_loss: %.4f, at_loss: %.4f, fm_loss: %.4f, kd_loss: %.4f, acc_1: \33[91m%.4f\033[0m, acc_5: %.4f" %
                                    ((epoch-1) * train_batch + step, loss_t, loss_ct, loss_ot, loss_pt, loss_at, loss_fm, loss_kd, acc_1t, acc_5t))

            acc1_reg = []
            acc5_reg = []
            loss_reg = []
            for step in range(valid_batch):
                batch_x = X_test[step * FLAGS.batch_size: (step + 1) * FLAGS.batch_size]
                batch_y = Y_test[step * FLAGS.batch_size: (step + 1) * FLAGS.batch_size]
                loss_v, acc_1v, acc_5v, summary = sess.run([loss, acc_top1, acc_top5, summary_op],
                                                        feed_dict={input_images: batch_x,
                                                                    y_true: batch_y,
                                                                    keep_prob: 1.,
                                                                    learning_rate: lr,
                                                                    clip_value: clipvalue})
                valid_writer.add_summary(summary, step + valid_batch * (epoch-1))
                acc1_reg.append(acc_1v)
                acc5_reg.append(acc_5v)
                loss_reg.append(loss_v)
            avg_acc1 = np.mean(np.array(acc1_reg))
            avg_acc5 = np.mean(np.array(acc5_reg))
            avg_loss = np.mean(np.array(loss_reg))
            print("Valid_loss ----> %0.4f Valid_acc ----> %0.4f, %0.4f" % (avg_loss, avg_acc1, avg_acc5))
            """
            " Save the best model
            """
            if avg_acc1 > best_acc:
                best_acc = avg_acc1
                saver_s.save(sess=sess,
                        save_path=FLAGS.t_s_logs_dir,
                        global_step=epoch)
                print("Save the best model with val_acc %0.4f" % best_acc)
            else:
                print("Val_acc stay with val_acc %0.4f" % best_acc)

            if last_loss - avg_loss > 1e-4 :
                last_loss = avg_loss
                patience = 0
                print("Patience %i with updated val_loss %0.4f" % (patience, last_loss))
            else:
                patience = patience + 1
                print("Patience %i with stayed val_loss %0.4f" % (patience, last_loss))

            if patience >= FLAGS.patience:
                patience = 0
                last_loss = 10000
                current = current * 0.65
                clipvalue = clipvalue * 0.65
                print("Early Stop, update the learning rate as %0.4f" % lr)
            end_time = time()
            print("Epoch %i ----> Ended in %0.4f" % (epoch, end_time - start_time))
            train_writer.close()
            valid_writer.close()
        print("......Ended")

        print("Ending......")

if __name__ == "__main__":
    tf.app.run()

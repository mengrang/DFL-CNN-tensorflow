# -*- coding : utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append("E:\plant\plant_disease_recognition")
import tensorflow as tf
from tensorflow.contrib import slim
from resnet import resnet_v2
import os
import time
import numpy as np
from time import time
from dataset import reader
from tflearn.data_utils import shuffle
from models.dfb_teacher import teacher
from models.dfb_student import student
from dfb_opt import train
import dfb_utils
from PIL import Image

"""
" Log Configuration
"""
tf.app.flags.DEFINE_string(name="data_dir", default="E:\plant", help="The directory to the dataset.")

tf.app.flags.DEFINE_string(name="train_dir", default="AgriculturalDisease_trainingset", help="The directory to the dataset.")

tf.app.flags.DEFINE_string(name="test_dir", default="AgriculturalDisease_testA", help="The directory to the dataset.")

tf.app.flags.DEFINE_string(name="valid_dir", default="AgriculturalDisease_validationset", help="The directory to the dataset.")

tf.app.flags.DEFINE_string(name="teacher_logs_dir", default="t_logs", help="The directory to the pre-trained  teachernet weights.")

tf.app.flags.DEFINE_integer(name="batch_size", default=55, help="The number of samples in each batch.")

tf.app.flags.DEFINE_integer(name="num_class", default=61, help="The number of classes.")

tf.app.flags.DEFINE_integer(name="epoches", default=1000, help="The number of training epoch.") 

tf.app.flags.DEFINE_integer(name="verbose", default=8, help="The number of training step to show the loss and accuracy.")

tf.app.flags.DEFINE_integer(name="patience", default=2, help="The patience of the early stop.")

tf.app.flags.DEFINE_boolean(name="debug", default=True, help="Debug mode or not")

tf.app.flags.DEFINE_float(name="alpha",default=0.5, help="The weight of knowledge distillation ")

tf.app.flags.DEFINE_float(name="beta", default=100.0, help="The weight of knowledge distillation ")

tf.app.flags.DEFINE_float(name="T", default=4.0, help="The temprature of knowledge distillation ")

tf.app.flags.DEFINE_float(name="Nratio",default=0.5, help="noisy ratio")

tf.app.flags.DEFINE_float(name="Nsigma",default=0.9, help="noisy sigma")

tf.app.flags.DEFINE_boolean(name="mimic",default=False, help="MImic mode or not")

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


def main(argv=None):
    is_training = True
    input_images = tf.placeholder(dtype=tf.float32,
                                      shape=[FLAGS.batch_size, 448, 448, 3],
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
    t_co_list, t_g, t_at, fc_obj, fc_part, fc_ccp, base_var_list, t_var_list = teacher(input_images, keep_prob, is_training=is_training)
    # Predictions 
    t_predictions = (tf.nn.softmax(fc_part) + 0.1 * tf.nn.softmax(fc_ccp) + tf.nn.softmax(fc_obj)) / 3.
    
    """
    ""teachernet loss
    """
    if not FLAGS.mimic:
        obj_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=fc_obj))
        part_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=fc_part))
        ccp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=fc_ccp))
        loss = 0.1 * ccp_loss + part_loss + obj_loss
        acc_top1 = dfb_utils.accuracy_top1(y_true, t_predictions)
        acc_top5 = dfb_utils.accuracy_top5(y_true, t_predictions)
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
    " Loading Data
    """
    print("Loading Data......")
    if FLAGS.mode == 'train':
        X_train, Y_train = reader.data_reader(FLAGS.data_dir, FLAGS.train_dir, 'AgriculturalDisease_train_pad_annotations.json')
        print(len(X_train))
        print(len(Y_train))
        print("\tLoaded Train Data......")

    X_test, Y_test = reader.data_reader(FLAGS.data_dir, FLAGS.valid_dir, 'AgriculturalDisease_validation_pad_annotations.json')  
    print(len(X_test))
    print(len(Y_test))
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
    saver_t = tf.train.Saver(t_var_list, max_to_keep=3)
    saver_base = tf.train.Saver(base_var_list,max_to_keep=3)
    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.teacher_logs_dir, 'train'),
                                        sess.graph)
   
    valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.teacher_logs_dir, 'valid'),
                                        sess.graph)

    
    print("Initialize global variables")                                   
    sess.run(tf.global_variables_initializer())

    """
    " Resume
    """
    ckpt_t = tf.train.get_checkpoint_state(FLAGS.teacher_logs_dir)
    if ckpt_t and ckpt_t.model_checkpoint_path:
        print('teacher:', ckpt_t.model_checkpoint_path)
        saver_t.restore(sess, ckpt_t.model_checkpoint_path)
        saver_base = tf.train.Saver(base_var_list,max_to_keep=3)
        print("Model restored...")
    ckpt_base = tf.train.get_checkpoint_state(FLAGS.teacher_logs_dir + '/base')
    if ckpt_base and ckpt_base.model_checkpoint_path:
        print('base:', ckpt_t.model_checkpoint_path)
        # saver_t.restore(sess, ckpt_t.model_checkpoint_path)
        saver_base.restore(sess, ckpt_base.model_checkpoint_path)
        print("Model restored...")

    """
    " Training...
    """
    if FLAGS.mode == 'train':                              
        train_batch = int(len(X_train) / FLAGS.batch_size)
        valid_batch = int(len(X_test) / FLAGS.batch_size)
        last_loss = 10000.
        patience = 0
        best_acc = 0.0
        clipvalue = 1e-4     
        current = 1e-4

        for epoch in range(63, FLAGS.epoches if FLAGS.debug else 1):
            print("Epoch %i ----> Starting......" % epoch)      
            # X_train, Y_train = shuffle(X_train, Y_train)   
            start_time = time()        
            """
            " Build learning rate
            """
            if epoch <= 10:
                lr = 1e-3 / 10.0 * epoch
                current = lr
            elif epoch > 10:
                lr = current 
            for step in range(train_batch):
                batch_x = X_train[step * FLAGS.batch_size: (step + 1) * FLAGS.batch_size]
                batch_y = Y_train[step * FLAGS.batch_size: (step + 1) * FLAGS.batch_size]
                summary, _ = sess.run([summary_op, train_op],
                                    feed_dict={input_images: batch_x,
                                                y_true: batch_y,
                                                keep_prob: 0.3,
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
                                                keep_prob: 0.3,
                                                learning_rate: lr,
                                                clip_value: clipvalue})
                        print("Step %i, Loss %0.4f, ccp_loss %0.4f,obj_loss %0.4f,part_loss %0.4f, acc_1 %0.4f,acc_5 %0.4f, lr %.7f,clip:%.7f" %
                                    ((epoch-1) * train_batch + step, loss_t, loss_ct, loss_ot, loss_pt, acc_1t, acc_5t, lr, clipvalue))
                    
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
            if avg_acc1 - best_acc > 1e-4:
                best_acc = avg_acc1
                saver_t.save(sess=sess,
                        save_path=os.path.join(FLAGS.teacher_logs_dir, 'teacher'),
                        global_step=epoch)
                saver_base.save(sess=sess,
                        save_path=FLAGS.teacher_logs_dir + '/base',
                        global_step=epoch)
                print("Save the best model with val_acc %0.4f" % best_acc)
            else:
                patience = patience + 1
                print("Patience %i with stayed val_loss %0.4f" % (patience, last_loss))
                print("Val_acc stay with val_acc %0.4f" % best_acc)

            # if last_loss - avg_loss > 1e-4:
            #     last_loss = avg_loss
            #     patience = 0
            #     print("Patience %i with updated val_loss %0.4f" % (patience, last_loss))
            # else:
            #     patience = patience + 1
            #     print("Patience %i with stayed val_loss %0.4f" % (patience, last_loss))

            # if avg_acc1 > best_acc:
            #     best_acc = avg_acc1
            #     saver_t.save(sess=sess,
            #             save_path=os.path.join(FLAGS.teacher_logs_dir, 'teacher'),
            #             global_step=epoch)
            #     saver_base.save(sess=sess,
            #             save_path=FLAGS.teacher_logs_dir + '/base',
            #             global_step=epoch)
            #     print("Save the best model with val_acc %0.4f" % best_acc)
            # else:
            #     print("Val_acc stay with val_acc %0.4f" % best_acc)

            # if last_loss - avg_loss > 1e-4:
            #     last_loss = avg_loss
            #     patience = 0
            #     print("Patience %i with updated val_loss %0.4f" % (patience, last_loss))
            # else:
            #     patience = patience + 1
            #     print("Patience %i with stayed val_loss %0.4f" % (patience, last_loss))

            if patience >= FLAGS.patience:
                patience = 0
                last_loss = 10000
                current = current * 0.8
                clipvalue = clipvalue * 0.8
                print("Lr decay, update the learning rate when lr = %0.4f" % lr)
            end_time = time()
            print("Epoch %i ----> Ended in %0.4f" % (epoch, end_time - start_time))
            train_writer.close()
            valid_writer.close()
        print("......Ended")

        print("Ending......")
   
if __name__ == "__main__":
    tf.app.run()
# coding:utf-8
import tensorflow as tf
"""
" Log Configuration
"""
tf.app.flags.DEFINE_string(name="data_dir", default="", help="The directory to the dataset.")

tf.app.flags.DEFINE_string(name="train_dir", default="", help="The directory to the dataset.")

tf.app.flags.DEFINE_string(name="test_dir", default="", help="The directory to the dataset.")

tf.app.flags.DEFINE_string(name="valid_dir", default="", help="The directory to the dataset.")

tf.app.flags.DEFINE_string(name="logs_dir", default="")

tf.app.flags.DEFINE_integer(name="batch_size", default=55, help="The number of samples in each batch.")

tf.app.flags.DEFINE_integer(name="num_class", default=61, help="The number of classes.")

tf.app.flags.DEFINE_integer(name="epoches", default=1000, help="The number of training epoch.") 

tf.app.flags.DEFINE_integer(name="verbose", default=8, help="The number of training step to show the loss and accuracy.")

tf.app.flags.DEFINE_integer(name="patience", default=2, help="The patience of the early stop.")

tf.app.flags.DEFINE_boolean(name="debug", default=True, help="Debug mode or not")

tf.flags.DEFINE_string(name="mode", default="train", help="Mode train/ test/ visualize")
FLAGS = tf.app.flags.FLAGS
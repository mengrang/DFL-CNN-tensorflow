# -*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

from collections import namedtuple
import functools

import tensorflow as tf

slim = tf.contrib.slim

# _CONV_DEFS specifies the MobileNet body
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
InvertedResidual = namedtuple('InvertedResidual', ['kernel', 'stride', 'depth', 'num', 'bottle_depth']) 
_CONV_DEFS = [
    Conv(kernel=[7, 7], stride=2, depth=64),
    InvertedResidual(kernel=[3, 3], stride=2, depth=128, num=1, bottle_depth=64),
    InvertedResidual(kernel=[3, 3], stride=2, depth=256, num=3, bottle_depth=64),
    InvertedResidual(kernel=[3, 3], stride=2, depth=512, num=4, bottle_depth=128),
    InvertedResidual(kernel=[3, 3], stride=2, depth=1024, num=6, bottle_depth=256),
    
]

def subsample(inputs, factor, scope=None):   
  if factor == 1:
    return inputs
  else:
    return layers.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  if stride == 1:
    return layers_lib.conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride=1,
        rate=rate,
        padding='SAME',
        scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = array_ops.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return layers_lib.conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride=stride,
        rate=rate,
        padding='VALID',
        scope=scope)


@slim.add_arg_scope
def _inverted_residual_bottleneck(inputs, depth, stride, bottleneck_depth, scope=None):
  with tf.variable_scope(scope, 'InvertedResidual', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, 1, stride=stride, 
                            activation_fn=None, normalizer_fn=None, scope='shortcut')
    output = slim.conv2d(preact, bottleneck_depth, 1, stride=1,
                              activation_fn=None, normalizer_fn=None, scope='conv1')
    """
    slim.conv2d(inputs,num_outputs,kernel_size,stride=1, padding='SAME',data_format=None,rate=1,activation_fn=nn.relu,normalizer_fn=None,
          normalizer_params=None,weights_initializer=initializers.xavier_initializer(),weights_regularizer=None,
          biases_initializer=init_ops.zeros_initializer(),biases_regularizer=None,
          reuse=None,variables_collections=None,outputs_collections=None,trainable=True,scope=None)
    """
    # output = slim.conv2d(output, bottleneck_depth, 3, stride=stride,
    #                           activation_fn=None, normalizer_fn=None, scope='conv2')
    output = conv2d_same(
        output, bottleneck_depth, 3, stride, rate=1, scope='conv2')

    output = slim.conv2d(output, depth, 1, stride=1,
                              activation_fn=None, normalizer_fn=None, scope='conv3')

    output = shortcut + output

    return output



def resnet_v2_base(inputs,
                      final_endpoint='InvertedResidual_{}_{}'.format(1024, 5),
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      scope=None):
  
  depth = lambda d: max(int(d * depth_multiplier), min_depth)
  end_points = {}

  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  if conv_defs is None:
    conv_defs = _CONV_DEFS

  if output_stride is not None and output_stride not in [8, 16, 32]:
    raise ValueError('Only allowed output_stride values are 8, 16, 32.')

  with tf.variable_scope(scope, 'ResNetV2', [inputs]):
    with slim.arg_scope([slim.conv2d], padding='SAME'):  
      current_stride = 1
      # The atrous convolution rate parameter.
      rate = 1
      net = inputs
      for i, conv_def in enumerate(conv_defs):
        if output_stride is not None and current_stride == output_stride:
          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          layer_stride = 1
          layer_rate = rate
          rate *= conv_def.stride
        else:
          layer_stride = conv_def.stride
          layer_rate = 1
          current_stride *= conv_def.stride

        if isinstance(conv_def, Conv):
          end_point = 'Conv2d_%d' % i
          net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel,
                            stride=conv_def.stride,
                            normalizer_fn=slim.batch_norm,
                            # biases_initializer=None,
                            scope=end_point)
          end_points[end_point] = net
          if end_point == final_endpoint:
            return net, end_points

        elif isinstance(conv_def, InvertedResidual):
          for n in range(conv_def.num):
            end_point = 'InvertedResidual_{}_{}'.format(conv_def.depth, n)
            stride = conv_def.stride if n == 0 else 1
            net = _inverted_residual_bottleneck(net, depth(conv_def.depth), stride, conv_def.bottle_depth, scope=end_point)
            end_points[end_point] = net

            if end_point == final_endpoint:
              return net, end_points
        else:
          raise ValueError('Unknown convolution type %s for layer %d'
                           % (conv_def.ltype, i))
  raise ValueError('Unknown final endpoint %s' % final_endpoint)


def resnet_v2(inputs,
                 num_classes=1000,
                 dropout_keep_prob=0.997,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 reuse=None,
                 scope='ResNetV2'):
 
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                     len(input_shape))

  with tf.variable_scope(scope, 'ResNetV2', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = resnet_v2_base(inputs, scope=scope,
                                          min_depth=min_depth,
                                          depth_multiplier=depth_multiplier,
                                          conv_defs=conv_defs)
    
  return net, end_points


def wrapped_partial(func, *args, **kwargs):
  partial_func = functools.partial(func, *args, **kwargs)
  functools.update_wrapper(partial_func, func)
  return partial_func


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


def mobilenet_v2_arg_scope(is_training=True,
                           weight_decay=0.00004,
                           stddev=0.09,
                           regularize_depthwise=False):
  """Defines the default MobilenetV2 arg scope.

  Args:
    is_training: Whether or not we're training the model.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    regularize_depthwise: Whether or not apply regularization on depthwise.

  Returns:
    An `arg_scope` to use for the mobilenet v2 model.
  """
  batch_norm_params = {
      'is_training': is_training,
      'center': True,
      'scale': True,
      'decay': 0.997,
      'epsilon': 0.001,
  }

  # Set weight_decay for weights in Conv and DepthSepConv layers.
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc
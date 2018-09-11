"""MobileNet v2.

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and different
head (for example: embeddings, localization and classification).

As described in https://arxiv.org/abs/1801.04381.

  MobileNets: Inverted Residuals and Linear Bottlenecks: Mobile Networks for
    Classification, Detection and Segmentation
  Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

"""

# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow as tf

slim = tf.contrib.slim

# _CONV_DEFS specifies the MobileNet body
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
InvertedResidual = namedtuple('InvertedResidual', ['kernel', 'stride', 'depth', 'num', 't']) # t is the expension factor
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    InvertedResidual(kernel=[3, 3], stride=1, depth=16, num=1, t=1),
    InvertedResidual(kernel=[3, 3], stride=2, depth=24, num=2, t=6),
    InvertedResidual(kernel=[3, 3], stride=2, depth=32, num=3, t=6),
    InvertedResidual(kernel=[3, 3], stride=2, depth=64, num=4, t=6),
    InvertedResidual(kernel=[3, 3], stride=1, depth=96, num=3, t=6),
    InvertedResidual(kernel=[3, 3], stride=2, depth=160, num=3, t=6),
    InvertedResidual(kernel=[3, 3], stride=1, depth=320, num=1, t=6),
    Conv(kernel=[1, 1], stride=1, depth=1280)
]

@slim.add_arg_scope
def _inverted_residual_bottleneck(inputs, depth, stride, expand_ratio, scope=None):
  with tf.variable_scope(scope, 'InvertedResidual', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    output = slim.conv2d(inputs, expand_ratio*inputs.get_shape().as_list()[-1], 1, stride=1,
                              activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm, scope='conv')
    """
    slim.conv2d(inputs,num_outputs,kernel_size,stride=1, padding='SAME',data_format=None,rate=1,activation_fn=nn.relu,normalizer_fn=None,
          normalizer_params=None,weights_initializer=initializers.xavier_initializer(),weights_regularizer=None,
          biases_initializer=init_ops.zeros_initializer(),biases_regularizer=None,
          reuse=None,variables_collections=None,outputs_collections=None,trainable=True,scope=None)
    """
    output = slim.separable_conv2d(output, None, 3, depth_multiplier=1, stride=stride,
                              activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm, scope='depthwise')
    output = slim.conv2d(output, depth, 1, stride=1,
                              activation_fn=None, normalizer_fn=slim.batch_norm, scope='pointwise')

    if stride==1 and depth==depth_in:
      shortcut = inputs
      output = shortcut + output

    return output



def mobilenet_v2_base(inputs,
                      final_endpoint='Conv2d_8',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      scope=None):
  """Mobilenet v2.

  Constructs a Mobilenet v2 network from inputs to the given final endpoint.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_0', 'InvertedResidual_16_0', 'InvertedResidual_24_0', 'InvertedResidual_24_1',
      'InvertedResidual_32_0', 'InvertedResidual_32_1', 'InvertedResidual_32_2',
      'InvertedResidual_64_0', 'InvertedResidual_64_1', 'InvertedResidual_64_2', 'InvertedResidual_64_3',
      'InvertedResidual_96_0', 'InvertedResidual_96_1', 'InvertedResidual_96_2',
      'InvertedResidual_160_0', 'InvertedResidual_160_1', 'InvertedResidual_160_2', 
      'InvertedResidual_320_0', 'Conv2d_8']
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of the activation maps. Allowed values are 8 (accurate fully convolutional
      mode), 16 (fast fully convolutional mode), 32 (classification mode).
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0, or the target output_stride is not
                allowed.
  """
  depth = lambda d: max(int(d * depth_multiplier), min_depth)
  end_points = {}

  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  if conv_defs is None:
    conv_defs = _CONV_DEFS

  if output_stride is not None and output_stride not in [8, 16, 32]:
    raise ValueError('Only allowed output_stride values are 8, 16, 32.')

  with tf.variable_scope(scope, 'MobilenetV2', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
      # The current_stride variable keeps track of the output stride of the
      # activations, i.e., the running product of convolution strides up to the
      # current network layer. This allows us to invoke atrous convolution
      # whenever applying the next convolution would result in the activations
      # having output stride larger than the target output_stride.
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
                            scope=end_point)
          end_points[end_point] = net
          if end_point == final_endpoint:
            return net, end_points

        elif isinstance(conv_def, InvertedResidual):
          for n in range(conv_def.num):
            end_point = 'InvertedResidual_{}_{}'.format(conv_def.depth, n)
            stride = conv_def.stride if n == 0 else 1
            net = _inverted_residual_bottleneck(net, depth(conv_def.depth), stride, conv_def.t, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
              return net, end_points
        else:
          raise ValueError('Unknown convolution type %s for layer %d'
                           % (conv_def.ltype, i))
  raise ValueError('Unknown final endpoint %s' % final_endpoint)


def mobilenet_v2(inputs,
                 num_classes=1000,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='MobilenetV2',
                 global_pool=False):
  """Mobilenet v2 model for classification.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    dropout_keep_prob: the percentage of activation values that are retained.
    is_training: whether is training or not.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    global_pool: Optional boolean flag to control the avgpooling before the
      logits layer. If false or unset, pooling is done with a fixed window
      that reduces default-sized inputs to 1x1, while larger inputs lead to
      larger outputs. If true, any input size is pooled down to 1x1.

  Returns:
    net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped-out input to the logits layer
      if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: Input rank is invalid.
  """
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                     len(input_shape))

  with tf.variable_scope(scope, 'MobilenetV2', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = mobilenet_v2_base(inputs, scope=scope,
                                          min_depth=min_depth,
                                          depth_multiplier=depth_multiplier,
                                          conv_defs=conv_defs)
      with tf.variable_scope('Logits'):
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
          end_points['global_pool'] = net
        else:
          # Pooling with a fixed kernel size.
          kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
          net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                scope='AvgPool_1a')
          end_points['AvgPool_1a'] = net
        if not num_classes:
          return net, end_points
        # 1 x 1 x 1024
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
      end_points['Logits'] = logits
      if prediction_fn:
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points

mobilenet_v2.default_image_size = 224


def wrapped_partial(func, *args, **kwargs):
  partial_func = functools.partial(func, *args, **kwargs)
  functools.update_wrapper(partial_func, func)
  return partial_func


mobilenet_v2_075 = wrapped_partial(mobilenet_v2, depth_multiplier=0.75)
mobilenet_v2_050 = wrapped_partial(mobilenet_v2, depth_multiplier=0.50)
mobilenet_v2_025 = wrapped_partial(mobilenet_v2, depth_multiplier=0.25)


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
      'decay': 0.9997,
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
                      activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc
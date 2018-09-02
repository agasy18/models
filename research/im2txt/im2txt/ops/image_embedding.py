# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image embedding ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base

slim = tf.contrib.slim


def inception_v3(images,
                 trainable=True,
                 is_training=True,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 add_summaries=True,
                 scope="InceptionV3"):
    """Builds an Inception V3 subgraph for image embeddings.

  Args:
    images: A float32 Tensor of shape [batch, height, width, channels].
    trainable: Whether the inception submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
      tf.contrib.layers.batch_norm for details.
    add_summaries: Whether to add activation summaries.
    scope: Optional Variable scope.

  Returns:
    end_points: A dictionary of activations from inception_v3 layers.
  """
    # Only consider the inception model to be in training mode if it's trainable.
    is_inception_model_training = trainable and is_training

    if use_batch_norm:
        # Default parameters for batch normalization.
        if not batch_norm_params:
            batch_norm_params = {
                "is_training": is_inception_model_training,
                "trainable": trainable,
                # Decay for the moving averages.
                "decay": 0.9997,
                # Epsilon to prevent 0s in variance.
                "epsilon": 0.001,
                # Collection containing the moving mean and moving variance.
                "variables_collections": {
                    "beta": None,
                    "gamma": None,
                    "moving_mean": ["moving_vars"],
                    "moving_variance": ["moving_vars"],
                }
            }
    else:
        batch_norm_params = None

    if trainable:
        weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        weights_regularizer = None

    with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_regularizer=weights_regularizer,
                trainable=trainable):
            with slim.arg_scope(
                    [slim.conv2d],
                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):
                net, end_points = inception_v3_base(images, scope=scope)
                with tf.variable_scope("logits"):
                    shape = net.get_shape()
                    net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                    net = slim.dropout(
                        net,
                        keep_prob=dropout_keep_prob,
                        is_training=is_inception_model_training,
                        scope="dropout")
                    net = slim.flatten(net, scope="flatten")

    # Add summaries.
    if add_summaries:
        for v in end_points.values():
            tf.contrib.layers.summaries.summarize_activation(v)

    return net


def ssd(images,
        trainable=True,
        is_training=True,
        weight_decay=0.00004,
        stddev=0.1,
        dropout_keep_prob=0.8,
        use_batch_norm=True,
        batch_norm_params=None,
        add_summaries=True,
        scope="SSD"):
    def flat_tensor(t):
        return tf.reshape(t, [tf.shape(t)[0], -1])

    from os.path import join
    model_file = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
    feature_layers = ['FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6:0']
    if not os.path.isfile(model_file):
        from subprocess import call
        url = 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz'
        call(['wget', '-nc', url])
        tar = 'ssd_mobilenet_v1_coco_2017_11_17.tar.gz'
        call(['tar', '-xf', tar, '-C', './'])
    feature_selector = lambda f: tf.concat([flat_tensor(n) for n in f], axis=1, name='selected_features')
    images = tf.cast((images + 1.0) * (0.5 * 255), dtype=tf.uint8, name='detector_image')

    with tf.gfile.GFile(model_file, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def = tf.GraphDef()
        od_graph_def.ParseFromString(serialized_graph)
        res = tf.import_graph_def(od_graph_def,
                                  name='',
                                  input_map={'image_tensor:0': images},
                                  return_elements=feature_layers)

    batch_size = 32

    FLAGS = tf.flags.FLAGS

    if hasattr(FLAGS, 'input_files'):
        batch_size = 1

    selected_features = tf.reshape(res[0], [batch_size, 19, 19, 512])

    def _depthwise_separable_conv(inputs,
                                  num_pwc_filters,
                                  width_multiplier,
                                  sc,
                                  downsample=False):
        """ Helper function to build the depth-wise separable convolution layer.
      """
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=_stride,
                                                      depth_multiplier=1,
                                                      kernel_size=[3, 3],
                                                      scope=sc + '/depthwise_conv')

        bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')
        pointwise_conv = slim.convolution2d(bn,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')
        return bn

    width_multiplier = 1
    net = _depthwise_separable_conv(selected_features, 512, width_multiplier, sc='x_conv_ds_11')
    net = _depthwise_separable_conv(selected_features, 512, width_multiplier, sc='x_conv_ds_12')
    net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='x_conv_ds_13')
    net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='x_conv_ds_14')
    net = _depthwise_separable_conv(net, 2048, width_multiplier, downsample=True, sc='x_conv_ds_15')
    net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='x_conv_ds_16')
    net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='x_conv_ds_17')
    net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='x_conv_ds_18')
    net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='x_conv_ds_19')
    
    return tf.reshape(net, [batch_size, 4096], name='emb_f')

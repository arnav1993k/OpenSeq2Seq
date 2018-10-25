# This code is heavily based on the code from MLPerf
# https://github.com/mlperf/reference/tree/master/translation/tensorflow/transformer

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Define defaults for parameters

class LayerNormalization(tf.layers.Layer):
  """Applies layer normalization."""

  def __init__(self, hidden_size, layer_norm_type = "L2"):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size
    self.norm = layer_norm_type

  def build(self, _):

    self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                   initializer=tf.ones_initializer(dtype=self.dtype),
                                   dtype=self.dtype)
    self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                  initializer=tf.zeros_initializer(dtype=self.dtype),
                                  dtype=self.dtype)
    self.built = True

  # def call(self, x, epsilon=1e-6):
  #   dtype = x.dtype
  #   x = tf.cast(x=x, dtype=tf.float32)
  #   mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
  #   variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
  #   norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  #   result = norm_x * self.scale + self.bias
  #   return tf.cast(x=result, dtype=dtype)

  def call(self, x, epsilon=1e-6):
    if self.norm=="L2":
      mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
      x = x - mean
      variance = tf.reduce_mean(tf.square(x), axis=[-1], keepdims=True)
      norm_x = x * tf.rsqrt(variance + epsilon)
    else: # "L1"
      mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
      x = x - mean
      variance = tf.reduce_mean(tf.abs(x), axis=[-1], keepdims=True)
      norm_x = tf.div(x , variance + epsilon)
    result = norm_x * self.scale + self.bias
    return result

class PrePostProcessingWrapper(object):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, params, train):
    self.layer = layer
    self.postprocess_dropout = params["layer_postprocess_dropout"]
    self.train = train
    # Create normalization layer
    self.layer_norm_type = params.get("layer_norm_type", "L2")
    self.layer_norm = LayerNormalization(hidden_size=params["hidden_size"],
                                         layer_norm_type=self.layer_norm_type)

  def __call__(self, x, *args, **kwargs):
    # Preprocessing: apply layer normalization
    y = self.layer_norm(x)

    # Get layer output
    y = self.layer(y, *args, **kwargs)

    # Postprocessing: apply dropout and residual connection
    if self.train:
      y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
    return x + y
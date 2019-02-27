# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from open_seq2seq.utils.utils import mask_nans, deco_print
from .loss import Loss


def dense_to_sparse(dense_tensor, sequence_length):
  indices = tf.where(tf.sequence_mask(sequence_length))
  values = tf.gather_nd(dense_tensor, indices)
  shape = tf.shape(dense_tensor, out_type=tf.int64)
  return tf.SparseTensor(indices, values, shape)


class CTCLoss(Loss):
  """Implementation of the CTC loss."""
  @staticmethod
  def get_optional_params():
    return dict(Loss.get_optional_params(), **{
        'mask_nan': bool,
    })

  def __init__(self, params, model, name="ctc_loss"):
    """CTC loss constructor.

    See parent class for arguments description.

    Config parameters:

    * **mask_nan** (bool) --- whether to mask nans in the loss output. Defaults
      to True.
    """
    super(CTCLoss, self).__init__(params, model, name)
    self._mask_nan = self.params.get("mask_nan", True)
    # this loss can only operate in full precision
    # if self.params['dtype'] != tf.float32:
    #   deco_print("Warning: defaulting CTC loss to work in float32")
    self.params['dtype'] = tf.float32

  def _compute_loss(self, input_dict):
    source_logits = input_dict['noisy_output']['logits']
    tgt_logits = input_dict['target_output']['logits']
    prob_source = tf.nn.softmax(source_logits)
    prob_target = tf.nn.softmax(tgt_logits)
    total_loss = tf.distributions.kl_divergence(
                prob_source,
                prob_target,
                allow_nan_stats=True,
                name=None
                )


    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(total_loss)
    return avg_loss

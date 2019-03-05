# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from open_seq2seq.models.model import Model
from open_seq2seq.utils.utils import deco_print

class DenoiserModel(Model):
  """
  Standard encoder-decoder class with one encoder and one decoder.
  "encoder-decoder-loss" models should inherit from this class.
  """

  @staticmethod
  def get_required_params():
    return dict(Model.get_required_params(), **{
        'encoder': None,  # could be any user defined class
        'decoder': None,  # could be any user defined class
        'denoiser':None,
    })

  @staticmethod
  def get_optional_params():
    return dict(Model.get_optional_params(), **{
        'encoder_params': dict,
        'decoder_params': dict,
        'denoiser_params':dict,
        'loss': None,  # could be any user defined class
        'loss_params': dict,
    })

  def __init__(self, params, mode="train", hvd=None):
    """Encoder-decoder model constructor.
    Note that TensorFlow graph should not be created here. All graph creation
    logic is happening inside
    :meth:`self._build_forward_pass_graph() <_build_forward_pass_graph>` method.

    Args:
      params (dict): parameters describing the model.
          All supported parameters are listed in :meth:`get_required_params`,
          :meth:`get_optional_params` functions.
      mode (string, optional): "train", "eval" or "infer".
          If mode is "train" all parts of the graph will be built
          (model, loss, optimizer).
          If mode is "eval", only model and loss will be built.
          If mode is "infer", only model will be built.
      hvd (optional): if Horovod is used, this should be
          ``horovod.tensorflow`` module.
          If Horovod is not used, it should be None.

    Config parameters:

    * **encoder** (any class derived from
      :class:`Encoder <encoders.encoder.Encoder>`) --- encoder class to use.
    * **encoder_params** (dict) --- dictionary with encoder configuration. For
      complete list of possible parameters see the corresponding class docs.
    * **decoder** (any class derived from
      :class:`Decoder <decoders.decoder.Decoder>`) --- decoder class to use.
    * **decoder_params** (dict) --- dictionary with decoder configuration. For
      complete list of possible parameters see the corresponding class docs.
    * **loss** (any class derived from
      :class:`Loss <losses.loss.Loss>`) --- loss class to use.
    * **loss_params** (dict) --- dictionary with loss configuration. For
      complete list of possible parameters see the corresponding class docs.
    """
    super(DenoiserModel, self).__init__(params=params, mode=mode, hvd=hvd)

    if 'encoder_params' not in self.params:
      self.params['encoder_params'] = {}
    if 'decoder_params' not in self.params:
      self.params['decoder_params'] = {}
    if 'loss_params' not in self.params:
      self.params['loss_params'] = {}
    data_layer = self.get_data_layer()
    self.params['decoder_params']['tgt_vocab_size'] = (
      data_layer.params['tgt_vocab_size']
    )
    self._encoder = self._create_encoder()
    self._decoder = self._create_decoder()
    self.denoiser = self._create_denoiser()
    if self.mode == 'train' or self.mode == 'eval':
      self._loss_computator = self._create_loss()
    else:
      self._loss_computator = None

  def _create_denoiser(self):
    """This function should return encoder class.
    Overwrite this function if additional parameters need to be specified for
    encoder, besides provided in the config.

    Returns:
      instance of a class derived from :class:`encoders.encoder.Encoder`.
    """
    params = self.params['denoiser_params']
    return self.params['denoiser'](params=params, mode=self.mode, model=self)
  def _create_encoder(self):
    """This function should return encoder class.
    Overwrite this function if additional parameters need to be specified for
    encoder, besides provided in the config.

    Returns:
      instance of a class derived from :class:`encoders.encoder.Encoder`.
    """
    params = self.params['encoder_params']
    return self.params['encoder'](params=params, mode='eval', model=self)

  def _create_decoder(self):
    """This function should return decoder class.
    Overwrite this function if additional parameters need to be specified for
    decoder, besides provided in the config.

    Returns:
      instance of a class derived from :class:`decoders.decoder.Decoder`.
    """
    params = self.params['decoder_params']
    return self.params['decoder'](params=params, mode='eval', model=self)

  def _create_loss(self):
    """This function should return loss class.
    Overwrite this function if additional parameters need to be specified for
    loss, besides provided in the config.

    Returns:
      instance of a class derived from :class:`losses.loss.Loss`.
    """
    return self.params['loss'](params=self.params['loss_params'], model=self)

  def _build_forward_pass_graph(self, input_tensors, gpu_id=0):
    """TensorFlow graph for encoder-decoder-loss model is created here.
    This function connects encoder, decoder and loss together. As an input for
    encoder it will specify source tensors (as returned from
    the data layer). As an input for decoder it will specify target tensors
    as well as all output returned from encoder. For loss it
    will also specify target tensors and all output returned from
    decoder. Note that loss will only be built for mode == "train" or "eval".

    Args:
      input_tensors (dict): ``input_tensors`` dictionary that has to contain
          ``source_tensors`` key with the list of all source tensors, and
          ``target_tensors`` with the list of all target tensors. Note that
          ``target_tensors`` only need to be provided if mode is
          "train" or "eval".
      gpu_id (int, optional): id of the GPU where the current copy of the model
          is constructed. For Horovod this is always zero.

    Returns:
      tuple: tuple containing loss tensor as returned from
      ``loss.compute_loss()`` and list of outputs tensors, which is taken from
      ``decoder.decode()['outputs']``. When ``mode == 'infer'``, loss will
      be None.
    """
    if not isinstance(input_tensors, dict) or \
       'source_tensors' not in input_tensors:
      raise ValueError('Input tensors should be a dict containing '
                       '"source_tensors" key')

    if not isinstance(input_tensors['source_tensors'], list):
      raise ValueError('source_tensors should be a list')

    source_tensors = input_tensors['source_tensors']
    if self.mode == "train" or self.mode == "eval":
      if 'target_tensors' not in input_tensors:
        raise ValueError('Input tensors should contain "target_tensors" key'
                         'when mode != "infer"')
      if not isinstance(input_tensors['target_tensors'], list):
        raise ValueError('target_tensors should be a list')
      target_tensors = input_tensors['target_tensors']

    with tf.variable_scope("ForwardPass",reuse=tf.AUTO_REUSE):
      denoiser_input = {"source_tensors": source_tensors}
      denoiser_output = self.denoiser(input_dict=denoiser_input)
      jasper_enc_denoised_input = {"source_tensors": denoiser_output}
      jasper_enc_denoised_output = self.encoder.encode(input_dict=jasper_enc_denoised_input)
      decoder_denoised_input = {"encoder_output": jasper_enc_denoised_output}
      jasper_denoised_output = self.decoder.decode(input_dict=decoder_denoised_input)
      if self.mode == "train" or self.mode == "eval":
        jasper_enc_target_input = {"source_tensors": target_tensors}
        jasper_enc_target_output = self.encoder.encode(input_dict=jasper_enc_target_input)
        decoder_target_input = {"encoder_output": jasper_enc_target_output}
        jasper_target_output = self.decoder.decode(input_dict=decoder_target_input)

      if self.mode == "train" or self.mode == "eval":
        with tf.variable_scope("Loss"):
          loss_input_dict = {
              "noisy_output": jasper_denoised_output,
              "target_output": jasper_target_output,
          }
          loss = self.loss_computator.compute_loss(loss_input_dict)
      else:
        deco_print("Inference Mode. Loss part of graph isn't built.")
        loss = None
      return loss, denoiser_output

  @property
  def encoder(self):
    """Model encoder."""
    return self._encoder

  @property
  def decoder(self):
    """Model decoder."""
    return self._decoder

  @property
  def loss_computator(self):
    """Model loss computator."""
    return self._loss_computator

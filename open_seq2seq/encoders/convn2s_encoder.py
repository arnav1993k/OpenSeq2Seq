import tensorflow as tf

class Block(object):
    def __init__(self,inplanes, planes, stride=1,block_id=0):
        self.conv1 = self.conv3x3(inplanes, planes, name="Basic_Block_{}_conv3x3_1".format(block_id))
        self.conv2 = self.conv3x3(inplanes, planes, name="Basic_Block_{}_conv3x3_2".format(block_id))
        self.conv3 = self.conv3x3(inplanes, planes, name="Basic_Block_{}_conv3x3_3".format(block_id))
        self.bn1 = tf.layers.BatchNormalization(axis=1)
        self.bn2 = tf.layers.BatchNormalization(axis=1)
        self.bn3 = tf.layers.BatchNormalization(axis=1)

    def __call__(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out_3x3 = self.conv3(x)
        out_3x3 = self.bn3(out_3x3)

        out = out + identity + out_3x3
        out = tf.nn.tanh(out)

        return out

    def conv1x1(self, in_planes, out_planes, stride=1, **kwargs):
        return tf.layers.conv2d(in_planes, out_planes, 1, use_bias=False, data_format='channels_first', **kwargs)

    def conv3x3(self, in_planes, out_planes, stride=1, **kwargs):
        return tf.layers.conv2d(in_planes, out_planes, 3, padding='same',use_bias=False, data_format='channels_first', **kwargs)


class Denoiser(object):
    def __init__(self,params, mode, model):
        self.channels = params["channels"]
        self.mode = True if mode in ["training","Training"] else False
        self.model = model

    def __call__(self,input_dict):
        x, src_length = input_dict['source_tensors']
        x = tf.expand_dims(x,axis=1)
        channels = self.channels
        out = self.conv1x1(x,channels[1],stride=1,name="Denoiser_layer_0")
        block_id = 1
        for filters in self.channels[2:-1]:
            out = self.block(out,filters,block_id,training=self.mode)
            block_id+=1
        out = self.conv3x3(out,channels[-1],name="Denoiser_layer_{}".format(block_id))
        out = tf.clip_by_value(out,-5,3)
        out = tf.squeeze(out,axis=1)
        return [out,src_length]
    def conv1x1(self, in_planes, out_planes, stride=1,**kwargs):
        return tf.layers.conv2d(in_planes,out_planes,1,use_bias=False,data_format='channels_first',**kwargs)
    def conv3x3(self, in_planes, out_planes, stride=1, **kwargs):
        return tf.layers.conv2d(in_planes,out_planes,3, padding='same',use_bias=False,data_format='channels_first',**kwargs)
    def block(self,x,filters,block_id,**kwargs):
        identity = x

        out = self.conv3x3(x,filters,name="Basic_Block_{}_conv3x3_1".format(block_id))
        out = tf.layers.batch_normalization(out,axis=1,name="Basic_Block_{}_bn_1".format(block_id),training=self.mode)
        out = tf.nn.relu(out,name="Basic_Block_{}_relu_1".format(block_id))
        out = self.conv3x3(out,filters,name="Basic_Block_{}_conv3x3_2".format(block_id))
        out = tf.layers.batch_normalization(out,axis=1,name="Basic_Block_{}_bn_2".format(block_id),training=self.mode)


        out = out + identity
        out = tf.nn.relu(out,name="Basic_Block_{}_relu_1".format(block_id))

        return out
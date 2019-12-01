import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, Lambda, LeakyReLU, \
    Conv2DTranspose

class Generator(tf.keras.Model):
    """
    Class for the pix2pix generator.
    """
    def __init__(self):
        """
        Definition of Generator model.
        Has a number of layers, including several convolutional layers, batch normalization,
        final dense layer, etc.
        """
        super(Generator, self).__init__()
        #TODO Add series of layers using tf.keras.layers
        pass


    @tf.function
    def call(self, images):
        """
        Applies Generator to batch of images
        :param outlines: Outlines of images. Size = [batch_size,height,width,outline_channels]
        :param images: Set of images corresponding to outlines. Either real images associated with outlines or outputs
        of generator when applied to outlines. Size = [batch_size,height,width,image_channels]
        :return: Set of probabilities that each image is the real image corresponding to its corresponding outline.
        Size = [batch_size,1]
        """
        #TODO Contatenate inputs by final dimesnsion, channels = outline_channels + image_channels

        #TODO Apply series of convolutional + dense layers. Use leaky relu

        #TODO Apply sigmoid to results
        pass


    @tf.function
    def loss_function(self,real_outputs,generated_outputs):
        """
        Calculates loss function, when real_outputs are probabilities corresponding to real images,
        :param real_outputs: Output of call when applied to real images.
        :param fake_outputs: Output of call when applied to outputs from generator.
        :return:
        """
        #TODO Add loss function, as defined in paper.
        pass

# Code adapted from:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class UnetSkipConnectionBlock(Layer):
    def __init__(self, inner_nc, outer_nc, submodule=None, outermost=False):
        Super(super(MyLayer, self).__init__()
        innermost = submodule is None
        assert(not (innermost and outermost))
        self.outermost = outermost
        self.submodule = submodule if not innermost else Lambda(lambda x: x)
        
        self.downrelu = Lambda(lambda x: x)
        self.downconv = Conv2D(inner_nc, kernel_size=(4, 4), strides=(2, 2), padding="same")
        self.downnorm = Lambda(lambda x: x)

        self.uprelu = ReLU()
        self.upconv = Conv2dTranspose(outer_nc, kernel_size=(4, 4), strides=(2, 2), padding="same")
        self.upnorm = BatchNormalization()

        if outermost:
            self.upnorm = tf.math.tanh
        else:
            self.downrelu = LeakyReLU(0.2)
            if not innermost:
                self.downnorm = BatchNormalization()
        

    @tf.function
    def call(self, inputs):
        down = self.downnorm(self.downconv(self.downrelu(inputs)))
        outputs = self.upnorm(self.upconv(self.uprelu(self.submodule(down))))
        if self.outermost:
            return outputs
        return tf.concat((inputs, outputs), 1)

import numpy as np
import tensorflow as tf

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
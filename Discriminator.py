import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
"""
Class for discriminator.
"""
class Discriminator(tf.keras.Model):
    def __init__(self):
        """
        Definition of Discriminator model.
        Has a number of layers, including several convolutional layers, batch normalization,
        final dense layer, etc.
        """
        super(Discriminator, self).__init__()
        #TODO Add series of layers using tf.keras.layers
        pass


    @tf.function
    def call(self,outlines,images):
        """
        Applies Discriminator to batch of images
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
    def loss_function(self,real_outputs,fake_outputs):
        """
        Calculates loss function, when real_outputs are probabilities corresponding to real images,
        :param real_outputs: Output of call when applied to real images.
        :param fake_outputs: Output of call when applied to outputs from generator.
        :return:
        """
        #TODO Add loss function, as defined in paper.
        pass
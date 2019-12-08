import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization
import numpy as np

k_init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
g_init = tf.keras.initializers.TruncatedNormal(mean=1.0, stddev=0.02)

# Code adapted from the official Torch implementat of pix2pix:
# https://github.com/phillipi/pix2pix/blob/master/models.lua
class PixelGAN(tf.keras.Model):
    def __init__(self, input_nc, output_nc, ndf=64):
        super(PixelGAN, self).__init__()
        self.conv_1 = Conv2D(
            filters=ndf, kernel_size=(1, 1), strides=(1, 1), padding="valid",
            kernel_initializer=k_init, input_shape=(256, 256, input_nc + output_nc))
        
        self.conv_2 = Conv2D(
            filters=ndf * 2, kernel_size=(1, 1), strides=(1, 1), padding="valid",
            kernel_initializer=k_init)
        self.batchnorm = BatchNormalization(gamma_initializer=g_init)

        self.conv_3 = Conv2D(
            filters=1, kernel_size=(1, 1), strides=(1, 1), padding="valid",
            kernel_initializer=k_init)

    @tf.function
    def call(self, input_, output):
        image_pairs = tf.concat([input_, output], -1)
        result = tf.nn.leaky_relu(self.conv_1(image_pairs), 0.2)
        result = tf.nn.leaky_relu(self.batchnorm(self.conv_2(result)), 0.2)
        return tf.math.reduce_mean(tf.math.sigmoid(self.conv_3(result)), axis=(1, 2, 3))

    @tf.function 
    def loss_function(self, disc_real_output, disc_fake_output):
        real_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(
            y_true=tf.ones_like(disc_real_output),
            y_pred=disc_real_output))
        fake_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(
            y_true=tf.zeros_like(disc_fake_output),
            y_pred=disc_fake_output))
        return real_loss + fake_loss

class Discriminator(tf.keras.Model):
    def __init__(self):
        """
        Definition of Discriminator model.
        Has a number of layers, including several convolutional layers, batch normalization,
        final dense layer, etc.
        """
        super(Discriminator, self).__init__()
        #TODO Add series of layers using tf.keras.layers
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="same")
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="same")
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding="same")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding="same")
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.conv5 = tf.keras.layers.Conv2D(512, kernel_size=(4, 4), padding="same")
        self.batch_norm4 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.compress = tf.keras.layers.Dense(1)
        self.optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)


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
        concatenated_inputs = tf.concat([outlines,images], axis = 3)
        #TODO Apply series of convolutional + dense layers. Use leaky relu
        conved_1 = tf.nn.leaky_relu(self.conv1(concatenated_inputs))
        conved_2 = self.conv2(conved_1)
        conved_2 = tf.nn.leaky_relu(self.batch_norm1(conved_2))
        conved_3 = self.conv3(conved_2)
        conved_3 = tf.nn.leaky_relu(self.batch_norm2(conved_3))
        conved_4 = self.conv4(conved_3)
        conved_4 = tf.nn.leaky_relu(self.batch_norm3(conved_4))
        conved_5 = self.conv4(conved_4)
        conved_5 = tf.nn.leaky_relu(self.batch_norm4(conved_5))
        flattened = self.flatten(conved_5)
        to_sigmoid = self.compress(flattened)
        outputs = tf.math.sigmoid(to_sigmoid)
        return outputs

    @tf.function 
    def loss_function(self, disc_real_output, disc_fake_output):
        real_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(
            y_true=tf.ones_like(disc_real_output),
            y_pred=disc_real_output))
        fake_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(
            y_true=tf.zeros_like(disc_fake_output),
            y_pred=disc_fake_output))
        return real_loss + fake_loss

class PatchGAN(tf.keras.Model):
    def __init__(self):
        """
        Definition of patchgan discriminator model.
        Has a number of layers, including several convolutional layers, batch normalization,
        final dense layer, etc.
        """
        super(PatchGAN, self).__init__()
        #TODO Add series of layers using tf.keras.layers
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="same")
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="same")
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding="same")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(512, kernel_size=(4, 4), strides=(1, 1), padding = "same")
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.conv5 = tf.keras.layers.Conv2D(1, kernel_size=(4, 4), strides = (1,1))
        self.optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)


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
        concatenated_inputs = tf.concat([outlines,images], axis = 3)
        #TODO Apply series of convolutional + dense layers. Use leaky relu
        conved_1 = tf.nn.leaky_relu(self.conv1(concatenated_inputs))
        conved_2 = self.conv2(conved_1)
        conved_2 = tf.nn.leaky_relu(self.batch_norm1(conved_2))
        conved_3 = self.conv3(conved_2)
        conved_3 = tf.nn.leaky_relu(self.batch_norm2(conved_3))
        conved_4 = self.conv4(conved_3)
        conved_4 = tf.nn.leaky_relu(self.batch_norm3(conved_4))
        conved_5 = self.conv5(conved_4)
        patch_probs = tf.math.sigmoid(conved_5)
        outputs = tf.math.reduce_mean(patch_probs, axis = 0)
        return outputs

    @tf.function 
    def loss_function(self, disc_real_output, disc_fake_output):
        real_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(
            y_true=tf.ones_like(disc_real_output),
            y_pred=disc_real_output))
        fake_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(
            y_true=tf.zeros_like(disc_fake_output),
            y_pred=disc_fake_output))
        return real_loss + fake_loss

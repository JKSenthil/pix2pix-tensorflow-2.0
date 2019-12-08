import tensorflow as tf
from tensorflow.keras import Model
import numpy as np

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
    def loss_function(self, disc_real_outputs, disc_fake_outputs):
        """
        Calculates loss function, when real_outputs are probabilities corresponding to real images,
        :param real_outputs: Output of call when applied to real images.
        :param fake_outputs: Output of call when applied to outputs from generator.
        :return:
        """
        # Add loss function, as defined in paper.
        real_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(
            y_true=tf.ones_like(disc_real_outputs),
            y_pred=disc_real_outputs))
        fake_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(
            y_true=tf.zeros_like(disc_fake_outputs),
            y_pred=disc_fake_outputs))
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
    def loss_function(self, disc_real_outputs, disc_fake_outputs):
        """
        Calculates loss function, when real_outputs are probabilities corresponding to real images,
        :param real_outputs: Output of call when applied to real images.
        :param fake_outputs: Output of call when applied to outputs from generator.
        :return:
        """
        # Add loss function, as defined in paper.
        real_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(
            y_true=tf.ones_like(disc_real_outputs),
            y_pred=disc_real_outputs))
        fake_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(
            y_true=tf.zeros_like(disc_fake_outputs),
            y_pred=disc_fake_outputs))
        return real_loss + fake_loss

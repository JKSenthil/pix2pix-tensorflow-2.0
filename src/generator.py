import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, Lambda, LeakyReLU, \
    Conv2DTranspose, Dropout, Activation

k_init=tf.keras.initializers.TruncatedNormal(stddev=0.02)
g_init=tf.keras.initializers.TruncatedNormal(mean=1.0, stddev=0.02)
# Code adapted from the Convolutional Autoendoer lab:
# https://drive.google.com/drive/u/0/folders/1mWRiFZE2ClSbE4Fxp9FgWH-7bwP3lvxQ
class Conv_BatchNorm_ReLU(Layer):
    def __init__(self, filters, used_in_encoder, use_dropout=False, **kwargs):
        super(Conv_BatchNorm_ReLU, self).__init__(**kwargs)
        if used_in_encoder: 
            self.conv = Conv2D(
                filters=filters, kernel_size=(4, 4), strides=(2, 2), padding="same",
                kernel_initializer=k_init)
        else:
            self.conv = Conv2DTranspose(
                filters=filters, kernel_size=(4, 4), strides=(2, 2), padding="same",
                kernel_initializer=k_init)
        self.batchnorm = BatchNormalization(gamma_initializer=g_init)
        self.dropout = Dropout(0.5) if use_dropout else Lambda(lambda x: x)
        self.relu = LeakyReLU(0.2) if used_in_encoder else Activation("relu")

    @tf.function
    def call(self, inputs):
        return self.relu(self.dropout(self.batchnorm(self.conv(inputs))))

class Encoder(Layer):
    def __init__(self, ngf=64, **kwargs):
       super(Encoder, self).__init__(**kwargs)
       used_in_encoder = True
       self.conv_1 = Conv2D(
           filters=ngf, kernel_size=(4, 4), strides=(2, 2), padding="same",
           kernel_initializer=k_init)
       self.conv_2 = Conv_BatchNorm_ReLU(ngf * 2, used_in_encoder)
       self.conv_3 = Conv_BatchNorm_ReLU(ngf * 4, used_in_encoder)
       self.conv_4 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder)
       self.conv_5 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder)
       self.conv_6 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder)
       self.conv_7 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder)
       self.conv_8 = Conv2D(
           filters=ngf * 8, kernel_size=(4, 4), strides=(2, 2), padding="same",
           kernel_initializer=k_init)

    @tf.function
    def call(self, inputs):
        temp = self.conv_4(self.conv_3(self.conv_2(tf.nn.leaky_relu(self.conv_1(inputs), 0.2))))
        return tf.nn.relu(self.conv_8(self.conv_7(self.conv_6(self.conv_5(temp)))))

class Decoder(Layer):
    def __init__(self, output_nc, ngf=64, **kwargs):
       super(Decoder, self).__init__(**kwargs)
       used_in_encoder = False 
       self.conv_1 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder, use_dropout=True)
       self.conv_2 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder, use_dropout=True)
       self.conv_3 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder, use_dropout=True)
       self.conv_4 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder)
       self.conv_5 = Conv_BatchNorm_ReLU(ngf * 4, used_in_encoder)
       self.conv_6 = Conv_BatchNorm_ReLU(ngf * 2, used_in_encoder)
       self.conv_7 = Conv_BatchNorm_ReLU(ngf * 1, used_in_encoder)
       self.conv_8 = Conv2DTranspose(
            filters=output_nc, kernel_size=(4, 4), strides=(2, 2), padding="same",
            kernel_initializer=k_init)

    @tf.function
    def call(self, inputs):
        temp = self.conv_4(self.conv_3(self.conv_2(self.conv_1(inputs))))
        return tf.math.tanh(self.conv_8(self.conv_7(self.conv_6(self.conv_5(temp)))))

class AutoEncoder(tf.keras.Model):
    def __init__(self, input_nc, output_nc):
        """
        Definition of Generator model.
        Has a number of layers, including several convolutional layers, batch normalization,
        final dense layer, etc.
        """
        super(AutoEncoder, self).__init__()
        # Add series of layers using tf.keras.layers
        self.encoder = Encoder(input_shape=(256, 256, input_nc))
        self.decoder = Decoder(output_nc)

    @tf.function
    def call(self, images):
        temp = self.encoder(images)
        print(temp.shape)
        result = self.decoder(temp)
        print(result.shape)
        return result

    @tf.function
    def loss_function(self, disc_fake_output, generated_output, ground_truth):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(disc_fake_output), y_pred=disc_fake_outputs)) + \
            tf.math.scalar_mul(tf.constant(100.0), tf.norm(generated_output - ground_truth, ord=1))

# Code adapted from: 
# https://github.com/phillipi/pix2pix/blob/master/models.lua
class UnetGenerator(tf.keras.Model):
    """
    Class for the pix2pix generator.
    """
    def __init__(self, input_nc, output_nc, ngf=64):
        super(UnetGenerator, self).__init__()
        self.conv_1 = Conv2D(
           filters=ngf, kernel_size=(4, 4), strides=(2, 2), padding="same", input_shape=(256, 256, input_nc),
           kernel_initializer=k_init)
        self.conv_2 = Conv_BatchNorm_ReLU(ngf * 2, used_in_encoder=True)
        self.conv_3 = Conv_BatchNorm_ReLU(ngf * 4, used_in_encoder=True)
        self.conv_4 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=True)
        self.conv_5 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=True)
        self.conv_6 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=True)
        self.conv_7 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=True)
        self.conv_8 = Conv2D(
           filters=ngf * 8, kernel_size=(4, 4), strides=(2, 2), padding="same",
           kernel_initializer=k_init)

        self.deconv_8 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=False, use_dropout=True)
        self.deconv_7 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=False, use_dropout=True)
        self.deconv_6 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=False, use_dropout=True)
        self.deconv_5 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=False)
        self.deconv_4 = Conv_BatchNorm_ReLU(ngf * 4, used_in_encoder=False)
        self.deconv_3 = Conv_BatchNorm_ReLU(ngf * 2, used_in_encoder=False)
        self.deconv_2 = Conv_BatchNorm_ReLU(ngf * 1, used_in_encoder=False)
        self.deconv_1 = Conv2DTranspose(
            filters=output_nc, kernel_size=(4, 4), strides=(2, 2), padding="same",
            kernel_initializer=k_init)

    @tf.function
    def call(self, images):
        encoder_output_1 = tf.nn.leaky_relu(self.conv_1(images), 0.2)
        encoder_output_2 = self.conv_2(encoder_output_1)
        encoder_output_3 = self.conv_3(encoder_output_2)
        encoder_output_4 = self.conv_4(encoder_output_3)
        encoder_output_5 = self.conv_5(encoder_output_4)
        encoder_output_6 = self.conv_6(encoder_output_5)
        encoder_output_7 = self.conv_7(encoder_output_6)
        encoder_output_8 = tf.nn.relu(self.conv_8(encoder_output_7))

        decoder_output = encoder_output_8
        for deconv, encoder_output in zip(
            [self.deconv_8, self.deconv_7, self.deconv_6, self.deconv_5, self.deconv_4, self.deconv_3, self.deconv_2],
            [encoder_output_7, encoder_output_6, encoder_output_5, encoder_output_4, encoder_output_3, encoder_output_2, encoder_output_1]
        ):
            _decoder_output = deconv(decoder_output)
            print(_decoder_output.shape, encoder_output.shape)
            decoder_output = tf.concat((_decoder_output, encoder_output), -1)
        return tf.math.tanh(self.deconv_1(decoder_output))

    @tf.function
    def loss_function(self, disc_fake_output, generated_output, ground_truth):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(disc_fake_output), y_pred=disc_fake_outputs)) + \
            tf.math.scalar_mul(tf.constant(100.0), tf.norm(generated_output - ground_truth, ord=1))

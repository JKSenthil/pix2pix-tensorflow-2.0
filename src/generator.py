import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, Lambda, LeakyReLU, \
    Conv2DTranspose, Dropout

# Code adapted from the Convolutional Autoendoer lab:
# https://drive.google.com/drive/u/0/folders/1mWRiFZE2ClSbE4Fxp9FgWH-7bwP3lvxQ
class Conv_BatchNorm_RelU(Layer):
    def __init__(self, filters, used_in_encoder, use_dropout=False, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        if used_in_encoder: 
            self.conv = Conv2DTranspose(
                filters=filters, kernel_size=(4, 4), strides=(2, 2), padding="same")
        else:
            self.conv = Conv2D(
                filters=filters, kernel_size=(4, 4), strides=(2, 2), padding="same")
        self.batchnorm = BatchNormalization()
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
           filters=ngf, kernel_size=(4, 4), strides=(2, 2), padding="same")
       self.relu_1 = LeakyReLU(0.2)
       self.conv_2 = Conv_BatchNorm_ReLU(ngf * 2, used_in_encoder)
       self.conv_3 = Conv_BatchNorm_ReLU(ngf * 4, used_in_encoder)
       self.conv_4 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder)
       self.conv_5 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder)
       self.conv_6 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder)
       self.conv_7 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder)
       self.conv_8 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder)

    @tf.function
    def call(self, inputs):
        temp = self.conv_4(self.conv_3(self.conv_2(self.relu_1(self.conv_1(inputs)))))
        return self.conv_8(self.conv_7(self.conv_6(self.conv_5(temp))))

class Decoder(Layer):
    def __init__(self, ngf=64, **kwargs):
       super(Encoder, self).__init__(**kwargs)
       used_in_encoder = True
       self.conv_1 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder, use_dropout=True)
       self.conv_2 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder, use_dropout=True)
       self.conv_3 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder, use_dropout=True)
       self.conv_4 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder)
       self.conv_5 = Conv_BatchNorm_ReLU(ngf * 4, used_in_encoder)
       self.conv_6 = Conv_BatchNorm_ReLU(ngf * 2, used_in_encoder)
       self.conv_7 = Conv_BatchNorm_ReLU(ngf * 1, used_in_encoder)

    @tf.function
    def call(self, inputs):
        temp = self.conv_4(self.conv_3(self.conv_2(self.conv_1(inputs))))
        return self.conv_7(self.conv_6(self.conv_5(temp)))

class AutoEncoder(tf.keras.Model):
    def __init__(self):
        """
        Definition of Generator model.
        Has a number of layers, including several convolutional layers, batch normalization,
        final dense layer, etc.
        """
        super(AutoEncoder, self).__init__()
        # Add series of layers using tf.keras.layers
        self.encoder = Encoder(input_shape=(256, 256, 3))
        self.decoder = Decoder()

    @tf.function
    def call(self, images):
        return self.decoder(self.encoder(images))

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

class UnetGenerator(tf.keras.Model):
    """
    Class for the pix2pix generator.
    """
    def __init__(self, ngf=64, lambda_=0.01, input_nc, output_nc):
        super(AutoEncoder, self).__init__()
        self.lambda_ = lambda_
        self.conv_1 = Conv_BatchNorm_ReLU(ngf * 1, used_in_encoder=True, input_shape=(256, 256, input_nc))
        self.conv_2 = Conv_BatchNorm_ReLU(ngf * 2, used_in_encoder=True)
        self.conv_3 = Conv_BatchNorm_ReLU(ngf * 4, used_in_encoder=True)
        self.conv_4 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=True)
        self.conv_5 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=True)
        self.conv_6 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=True)
        self.conv_7 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=True)
        self.conv_8 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=True)

        self.deconv_8 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=False, use_dropout=True)
        self.deconv_7 = Conv_BatchNorm_ReLU(ngf * 16, used_in_encoder=False, use_dropout=True)
        self.deconv_6 = Conv_BatchNorm_ReLU(ngf * 16, used_in_encoder=False, use_dropout=True)
        self.deconv_5 = Conv_BatchNorm_ReLU(ngf * 16, used_in_encoder=False)
        self.deconv_4 = Conv_BatchNorm_ReLU(ngf * 16, used_in_encoder=False)
        self.deconv_3 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder=False)
        self.deconv_2 = Conv_BatchNorm_ReLU(ngf * 4, used_in_encoder=False)
        self.deconv_1 = Conv_BatchNorm_ReLU(output_nc, used_in_encoder=False)

    @tf.function
    def call(self, images):
        output_1 = self.conv_1(images)
        output_2 = self.conv_2(output_1)
        output_3 = self.conv_3(output_2)
        output_4 = self.conv_4(output_3)
        output_5 = self.conv_5(output_4)
        output_6 = self.conv_6(output_5)
        output_7 = self.conv_7(output_6)
        output_8 = self.conv_7(output_7)

        result = self.deconv_8(output_8)
        result = self.deconv_7(tf.concat([output_7, result], -1))
        result = self.deconv_6(tf.concat([output_6, result], -1))
        result = self.deconv_5(tf.concat([output_5, result], -1))
        result = self.deconv_4(tf.concat([output_4, result], -1))
        result = self.deconv_3(tf.concat([output_3, result], -1))
        result = self.deconv_2(tf.concat([output_2, result], -1))
        return self.deconv_1(tf.concat([output_1, result], -1))

    @tf.function
    def loss_function(self, discriminator, generated_outputs, ground_truth):
        disc_fake_outputs = discriminator(generated_outputs)
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=tf.zeros_like(logits_fake), y_pred=disc_fake_outputs)) + \
            tf.math.scalar_mul(tf.constant(self.lambda_), tf.norm(generated_outputs - ground_truth, 1))

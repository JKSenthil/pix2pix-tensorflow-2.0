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
       self.conv_1 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder, True)
       self.conv_2 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder, True)
       self.conv_3 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder, True)
       self.conv_4 = Conv_BatchNorm_ReLU(ngf * 8, used_in_encoder)
       self.conv_5 = Conv_BatchNorm_ReLU(ngf * 4, used_in_encoder)
       self.conv_6 = Conv_BatchNorm_ReLU(ngf * 2, used_in_encoder)
       self.conv_7 = Conv_BatchNorm_ReLU(ngf * 1, used_in_encoder)

    @tf.function
    def call(self, inputs):
        temp = self.conv_4(self.conv_3(self.conv_2(self.conv_1(inputs))))
        return self.conv_7(self.conv_6(self.conv_5(temp)))

class AutoEncoder(tf.keras.Model):
    """
    Class for the pix2pix generator.
    """
    def __init__(self):
        """
        Definition of Generator model.
        Has a number of layers, including several convolutional layers, batch normalization,
        final dense layer, etc.
        """
        super(AutoEncoder, self).__init__()
        #TODO Add series of layers using tf.keras.layers
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
    def __init__(self, ngf=64):
        super(UnetGenerator, self).__init__()
        self.unet = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_shape=(256, 256, 3))
        for _ in range(4):
            self.unet = UnetSkipConnectionBlock(ngf * 8, ngf * 16, submodule=self.unet)
        self.unet = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=self.unet)
        self.unet = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=self.unet)
        self.unet = UnetSkipConnectionBlock(ngf * 1, ngf * 2, submodule=self.unet, outermost=True)

    @tf.function
    def call(self, images):
        return self.unet(images)

    @tf.function
    def loss_function(self,real_outputs,generated_outputs):
        pass

# Code adapted from:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class UnetSkipConnectionBlock(Layer):
    def __init__(self, inner_nc, outer_nc, submodule=None, outermost=False, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
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
        return tf.concat((inputs, outputs), -1)

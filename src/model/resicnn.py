from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error
import tensorflow as tf
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.merge import Add, Subtract
import json
import os
from keras import backend as K


class ResiCNN:  # My JackNet with residual block
    cnn_filter_num = 64
    cnn_kernel_size = 3

    def __init__(self, channels=3):
        self.model = None
        self.optimizer = None
        self.channels = channels

    def bulid(self):  # build model
        image_in = Input((None, None, self.channels))

        conv = Conv2D(filters=self.cnn_filter_num, kernel_size=self.cnn_kernel_size,
                       strides=(1, 1), padding='same', data_format='channels_last')(image_in)
        conv = Activation('relu')(conv)

        x = conv

        for layers in range(8):
            x = self._build_residual_block(x)

        conv_out = Conv2D(filters=self.channels, kernel_size=self.cnn_kernel_size,
                       strides=(1, 1), padding='same', data_format='channels_last')(x)

        output = Add()([image_in, conv_out])

        self.model = Model(image_in, output, name='model')

    def _build_residual_block(self, x): # build residual block
        x_in = x

        x = Conv2D(filters=self.cnn_filter_num,
                   kernel_size=self.cnn_kernel_size,
                   strides=(1, 1), padding='same', data_format='channels_last')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=self.cnn_filter_num,
                   kernel_size=self.cnn_kernel_size,
                   strides=(1, 1), padding='same', data_format='channels_last')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x_in, x])
        x = Activation("relu")(x)
        return x

    def predict(self, x): # denoise on input x
        if x.ndim == 3:
            x = x.reshape(1, x.shape[0], x.shape[1], self.channels)
        return self.model.predict_on_batch(x)

    def load(self, config_path, model_path): # load model
        print('restore model...')
        if os.path.exists(config_path) and os.path.exists(model_path):
            with open(config_path, 'r') as fp:
                self.model = Model.from_config(json.load(fp))
                self.model.load_weights(model_path)
            return True
        return False

    def save(self, config_path, model_path): # save model
        with open(config_path, 'w') as fp:
            json.dump(self.model.get_config(), fp)
            self.model.save_weights(model_path)

    def compile(self): # choose adam optimizer and set learning rate
        self.optimizer = Adam(lr=1e-2)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def train_generator(self, data, epochs=1, steps_per_epochs=None, callbacks=None):
        self.model.fit_generator(iter(data), epochs=epochs, steps_per_epoch=steps_per_epochs
                          , callbacks=callbacks)

    def train(self, data, epochs=1, callbacks=None):
        self.model.fit(x=data[0], y=data[1], epochs=epochs, batch_size=8
                          , callbacks=callbacks)

    @staticmethod
    def loss(y_true, y_pred):  # loss function, mean square error
        return 0.5 * K.sum(K.square(y_pred - y_true), axis=-1)
'''
Author: hanyu
Date: 2022-07-19 16:06:08
LastEditTime: 2022-08-23 15:58:26
LastEditors: hanyu
Description: cnn
FilePath: /RL_Lab/networks/cnn.py
'''
import tensorflow as tf

from configs.config_base import PolicyParams


def cnn_simple(inputs, params: PolicyParams, name: str = "cnn_simple"):
    last_layer = inputs
    for i, (filter, kernel, stride) in enumerate(params.cnn_filters[:-1]):
        last_layer = tf.keras.layers.Conv2D(
            filters=filter,
            kernel_size=kernel,
            strides=stride,
            padding='same',
            activation=params.activation,
            kernel_initializer=params.kernel_initializer,
            name=f"{name}_conv_{i}")(last_layer)

    outsize, kernel, stride = params.cnn_filters[-1]
    conv_out = tf.keras.layers.Conv2D(filters=outsize,
                                      kernel_size=kernel,
                                      strides=stride,
                                      activation=params.activation,
                                      padding="valid",
                                      name=f"{name}_conv_{i + 1}")(last_layer)
    # conv_out = tf.keras.layers.Flatten()(last_layer)

    # conv_out = tf.keras.layers.Dense(256)(conv_out)
    # conv_out = tf.keras.layers.Activation("relu",
    #                                       name=f"{name}_relu_4")(conv_out)

    return conv_out


def impala_cnn(inputs, params: PolicyParams, name="impala_cnn"):
    # # Initialize inputs layer
    # inputs = tf.keras.layers.Input(shape=(None, ) + tuple(params.input_shape),
    #                                name=f"{name}_obs")
    conv_out = inputs
    conv_out = tf.cast(conv_out, tf.float32)
    conv_out /= 255
    for i, (filter, kernel, stride,
            num_blocks) in enumerate(params.resnet_filters):
        # Downscale
        conv_out = tf.keras.layers.Conv2D(filters=filter,
                                          kernel_size=kernel,
                                          strides=stride,
                                          padding='same',
                                          activation=None,
                                          kernel_initializer=None,
                                          name=f"{name}_conv_{i}")(conv_out)
        conv_out = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                             padding='same',
                                             strides=(2, 2))(conv_out)

        # Residual blocks
        for j in range(num_blocks):
            block_input = conv_out
            conv_out = tf.keras.layers.Activation(
                "relu", name=f"{name}_relu_{i}_{j}_1")(conv_out)
            conv_out = tf.keras.layers.Conv2D(
                filters=filter,
                kernel_size=kernel,
                strides=stride,
                padding='same',
                activation=None,
                kernel_initializer=None,
                name=f"{name}_residual_conv_{i}_{j}_1")(conv_out)
            conv_out = tf.keras.layers.Activation(
                "relu", name=f"{name}_relu_{i}_{j}_2")(conv_out)
            conv_out = tf.keras.layers.Conv2D(
                filters=filter,
                kernel_size=kernel,
                strides=stride,
                padding='same',
                activation=None,
                kernel_initializer=None,
                name=f"{name}_residual_conv_{i}_{j}_2")(conv_out)
            conv_out = tf.keras.layers.Add(name=f"{name}_add_{i}_{j}")(
                [conv_out, block_input])

    # conv_out = tf.keras.layers.Activation("relu",
    #                                       name=f"{name}_relu_3")(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)
    conv_out = tf.keras.layers.Activation("relu",
                                          name=f"{name}_relu_3")(conv_out)
    conv_out = tf.keras.layers.Dense(units=256, activation="relu")(conv_out)
    # conv_out = tf.keras.layers.Activation("relu",
    #                                       name=f"{name}_relu_4")(conv_out)
    return conv_out

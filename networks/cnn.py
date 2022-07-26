'''
Author: hanyu
Date: 2022-07-19 16:06:08
LastEditTime: 2022-07-26 16:08:44
LastEditors: hanyu
Description: cnn
FilePath: /RL_Lab/networks/cnn.py
'''
import tensorflow as tf

from configs.config_base import PolicyParams


def cnn_simple(params: PolicyParams):
    model = tf.keras.Sequential(name='simple_cnn')
    for (filter, kernel, stride) in params.cnn_filters[:-1]:
        model.add(
            tf.keras.layers.Conv2D(
                filters=filter,
                kernel_size=kernel,
                strides=stride,
                padding='valid',
                activation=params.activation,
                kernel_initializer=params.kernel_initializer))
        model.add(tf.keras.layers.MaxPool2D(pool_size=params.pool_size))
    out_size, kernel, stride = params.cnn_filters[-1]
    model.add(tf.keras.layers.Flatten())
    model.add(
        tf.keras.layers.Dense(units=out_size,
                              activation=params.output_activation,
                              kernel_initializer=params.kernel_initializer))
    return model, out_size


def impala_cnn(inputs, params: PolicyParams, name="impala_cnn"):
    # # Initialize inputs layer
    # inputs = tf.keras.layers.Input(shape=(None, ) + tuple(params.input_shape),
    #                                name=f"{name}_obs")
    last_layer = inputs
    for i, (filter, kernel, stride,
            num_blocks) in enumerate(params.resnet_filters):
        # Downscale
        last_layer = tf.keras.layers.Conv2D(
            filters=filter,
            kernel_size=kernel,
            strides=stride,
            padding='valid',
            activation=params.activation,
            kernel_initializer=params.kernel_initializer,
            name=f"{name}_conv_{i}")(last_layer)
        last_layer = tf.keras.layers.MaxPool2D(padding='same',
                                               strides=(2, 2))(last_layer)

        # Residual blocks
        for j in range(num_blocks):
            block_input = last_layer
            conv_out = tf.keras.layers.Activation(
                "relu", name=f"{name}_relu_{i}_{j}_1")(block_input)
            conv_out = tf.keras.layers.Conv2D(
                filters=filter,
                kernel_size=kernel,
                strides=stride,
                padding='same',
                activation=params.activation,
                kernel_initializer=params.kernel_initializer,
                name=f"{name}_residual_conv_{i}_{j}_1")(conv_out)
            conv_out = tf.keras.layers.Activation(
                "relu", name=f"{name}_relu_{i}_{j}_2")(conv_out)
            conv_out = tf.keras.layers.Conv2D(
                filters=filter,
                kernel_size=kernel,
                strides=stride,
                padding='same',
                activation=params.activation,
                kernel_initializer=params.kernel_initializer,
                name=f"{name}_residual_conv_{i}_{j}_2")(conv_out)
            last_layer = tf.keras.layers.Add(name=f"{name}_add_{i}_{j}")(
                [block_input, conv_out])
        conv_out = tf.keras.layers.Activation("relu",
                                              name=f"{name}_relu_3")(conv_out)
    # TODO
    conv_out = tf.keras.layers.Flatten()(conv_out)

    conv_out = tf.keras.layers.Dense(256)(conv_out)
    conv_out = tf.keras.layers.Activation("relu",
                                          name=f"{name}_relu_4")(conv_out)
    return conv_out

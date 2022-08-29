'''
Author: hanyu
Date: 2021-08-12 03:03:52
LastEditTime: 2021-08-12 03:05:15
LastEditors: hanyu
Description: 
FilePath: /LordHD-RL/env/sl_helper/sl_policy_cls.py
'''
import math
import tensorflow.compat.v1 as tf


class Policy:

    def __init__(self) -> None:
        gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        config_tf = tf.ConfigProto(gpu_options=gpu_option)
        self.session = tf.Session(config=config_tf)

        self.batch_norm_count = 0
        self.block_num = 10
        self.filter_num = 128

        self.game_state = tf.placeholder(dtype=tf.float32)
        self.game_state1 = tf.transpose(self.game_state, [1, 0])

        self.move = self.resnet(self.game_state1)
        self.prob = tf.nn.softmax(self.move)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.session.run(self.init)

    def init_app(self, app):
        self.saver.restore(self.session, app.config['MODEL_PATH_POLICY'])

    def resnet(self, s):
        # NCHW format is faster in GPU with cuDNN, but is not available in CPU mode by now.
        s_planes = tf.reshape(s, [-1, 4, 15, 13])

        # Input block
        data_flow = self.conv_block(
            s_planes, filter_size=3, input_channels=13, output_channels=self.filter_num)

        # Residual tower
        for _ in range(self.block_num):
            data_flow = self.residual_block(data_flow)

        # Policy head
        conv_head = tf.reshape(
            self.conv_block(data_flow, filter_size=1, input_channels=self.filter_num, output_channels=2), [-1, 4 * 15 * 2])

        weight_fc = weight_variable_init([4 * 15 * 2, 309])
        bias_fc = bias_variable_init([309])
        fc_output = tf.add(tf.matmul(conv_head, weight_fc), bias_fc)

        return fc_output

    def get_batch_norm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1

        return result

    def conv_block(self, inputs, filter_size, input_channels, output_channels):
        weights = weight_variable_init(
            [filter_size, filter_size, input_channels, output_channels])
        bias = bn_bias_variable_init([output_channels])
        weight_key = self.get_batch_norm_key()
        with tf.variable_scope(weight_key, reuse=tf.AUTO_REUSE):
            output = batch_normalization(conv2d(inputs, weights), bias)

        return tf.nn.relu(output)

    def residual_block(self, inputs):
        origin_input = tf.identity(inputs)

        # First convolution with batch norm
        weights_1 = weight_variable_init(
            [3, 3, self.filter_num, self.filter_num])
        bias_1 = bn_bias_variable_init([self.filter_num])
        weight_key_1 = self.get_batch_norm_key()
        with tf.variable_scope(weight_key_1, reuse=tf.AUTO_REUSE):
            bn_output1 = batch_normalization(conv2d(inputs, weights_1), bias_1)

        output1 = tf.nn.relu(bn_output1)

        # Second convolution with batch norm
        weights_2 = weight_variable_init(
            [3, 3, self.filter_num, self.filter_num])
        bias_2 = bn_bias_variable_init([self.filter_num])
        weight_key_2 = self.get_batch_norm_key()
        with tf.variable_scope(weight_key_2, reuse=tf.AUTO_REUSE):
            bn_output2 = batch_normalization(
                conv2d(output1, weights_2), bias_2)

        # This ReLU is outside from addition.
        output2 = tf.nn.relu(tf.add(bn_output2, origin_input))
        return output2


def weight_variable_init(shape):
    # Xavier distribution
    stddev = math.sqrt(2.0 / sum(shape))
    return tf.Variable(tf.truncated_normal(shape, mean=0, stddev=stddev))


def bias_variable_init(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))


def bn_bias_variable_init(shape):
    # It is not trainable because of batch normalization.
    # Thus, parameters of network reduces, and training gets faster.
    return tf.Variable(tf.constant(0.0, shape=shape), trainable=False)


def conv2d(inputs, filters):
    return tf.nn.conv2d(inputs, filters, data_format='NHWC', strides=[1, 1, 1, 1], padding='SAME')


# 此处的train啊
def batch_normalization(conv, bias):
    return tf.layers.batch_normalization(
        tf.nn.bias_add(conv, bias, data_format='NHWC'),
        epsilon=1e-5, axis=-1, fused=False, center=False, scale=False)

'''
Author: hanyu
Date: 2020-11-02 09:51:50
LastEditTime: 2021-06-16 13:16:41
LastEditors: hanyu
Description: restnet with action masked
FilePath: /test_ppo/examples/PPO_hungry_geese/rllib_training/MaskedResNetClass.py
'''
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()
# print(f'TensorFlow Version is {tfv}')
# print(f'tf1 eager mode: {tf1.executing_eagerly()}')
# print(f'tf eager mode: {tf.executing_eagerly()}')


class MaskedResidualNetwork(TFModelV2):
    """Custom resnet model for ppo algorithm"""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(MaskedResidualNetwork, self).__init__(obs_space, action_space,
                                                    num_outputs, model_config, name)
        custom_model_config = model_config.get('custom_model_config')
        activation = 'relu'
        filters = custom_model_config.get("conv_filters", None)
        if filters is None:
            print('you must set the "conv_filters" in model config!')
            raise NotImplementedError
        no_final_linear = custom_model_config.get('no_final_linear')
        vf_share_layers = custom_model_config.get('vf_share_layers')

        inputs = tf.keras.layers.Input(
            shape=(custom_model_config.get('rows'), custom_model_config.get('columns'),
                   custom_model_config.get('channels')),
            name='observations'
        )
        last_layer = inputs

        # whether the last layer is the output of a Flattened
        self.last_layer_is_flattened = False

        # build the action layers
        for i, (out_size, kernel, stride, blocks) in enumerate(filters[:-1], 1):
            last_layer = self.residual_stack(x=last_layer,
                                             filters=out_size,
                                             blocks=blocks,
                                             name=f'stack{i}')

        out_size, kernel, stride, _ = filters[-1]

        # no final linear: Last Layer is a Conv2D and uses num_outputs.
        if no_final_linear and num_outputs:
            last_layer = tf.keras.layers.Conv2D(
                num_outputs,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding='valid',
                data_format='channels_last',
                name='conv_out'
            )(last_layer)
            conv_out = last_layer
        # finish network normally, then add another linear one of size `num_outputs`.
        else:
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding='valid',
                data_format='channels_last',
                name=f'conv{i + 1}'
            )(last_layer)

            # num_outputs defined. Use that to create an exact `num_output`-sized (1,1)-Conv2D.
            if num_outputs:
                conv_out = tf.keras.layers.Conv2D(
                    num_outputs,
                    [1, 1],
                    activation=None,
                    padding='same',
                    data_format='channels_last',
                    name='conv_out'
                )(last_layer)
            # num_outputs not known -> Flatten, then set self.num_outputs to the resulting number of nodes.
            else:
                self.last_layer_is_flattened = True
                conv_out = tf.keras.layers.Flatten(
                    data_format='channels_last'
                )(last_layer)
                self.num_outputs = conv_out.shape[1]

        # build the value layers
        if vf_share_layers:
            last_layer = tf.keras.layers.Lambda(
                lambda x: tf.squeeze(x, axis=[1, 2])
            )(last_layer)
            value_out = tf.keras.layers.Dense(
                1,
                name='value_out',
                activation=None,
                kernel_initializer=normc_initializer(0.01)
            )(last_layer)
        else:
            # build a parallel set of hidden layers for the value net
            last_layer = inputs
            for i, (out_size, kernel, stride, _) in enumerate(filters[:-1], 1):
                last_layer = tf.keras.layers.Conv2D(
                    out_size,
                    kernel,
                    strides=(stride, stride),
                    activation=activation,
                    padding='same',
                    data_format='channels_last',
                    name=f'conv_value_{i}'
                )(last_layer)

            out_size, kernel, stride, _ = filters[-1]
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding='valid',
                data_format='channels_last',
                name=f'conv_value_{i + 1}'
            )(last_layer)
            last_layer = tf.keras.layers.Conv2D(
                1,
                (1, 1),
                activation=None,
                padding='same',
                data_format='channels_last',
                name='conv_value_out'
            )(last_layer)
            value_out = tf.keras.layers.Lambda(
                lambda x: tf.squeeze(x, axis=[1, 2])
            )(last_layer)

        self.base_model = tf.keras.Model(inputs, [conv_out, value_out])
        # self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager.
        model_out, self._value_out = self.base_model(
            tf.cast(input_dict['obs']['state'], tf.float32)
        )

        # our last layer is already flat
        if self.last_layer_is_flattened:
            raise ValueError
            avail_actions = input_dict['obs']['avail_actions']
            action_mask = input_dict['obs']['action_mask']
            intent_vector = tf.expand_dims(model_out, 1)
            action_logits = tf.reduce_sum(
                avail_actions * intent_vector, axis=1)
            inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
            return action_logits + inf_mask, state
        # last layer is a n x [1, 1] Conv2D -> Flatten
        else:
            model_out = tf.squeeze(model_out, axis=[1, 2])
            avail_actions = input_dict['obs']['avail_actions']
            action_mask = input_dict['obs']['action_mask']
            intent_vector = tf.expand_dims(model_out, 1)
            action_logits = tf.math.reduce_sum(
                intent_vector, axis=1)
            inf_mask = tf.math.maximum(
                tf.math.log(action_mask), tf.float32.min)

            return action_logits + inf_mask, state
        # # explicit cast to float32 needed in eager.
        # model_out, self._value_out = self.base_model(
        #     tf.cast(input_dict['obs'], tf.float32)
        # )
        # # our last layer is already flat
        # if self.last_layer_is_flattened:
        #     return model_out, state
        # # last layer is a n x [1, 1] Conv2D -> Flatten
        # else:
        #     return tf.squeeze(model_out, axis=[1, 2]), state

    def save(self):
        self.base_model.save('ws.h5')

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def residual_stack(self,
                       x,
                       filters,
                       blocks,
                       name=None):

        x = self.residual_block(x=x,
                                filters=filters,
                                strides=(1, 1),
                                name=f'{name}_blockv1')
        for i in range(2, blocks + 1):
            x = self.residual_block(x=x,
                                    filters=filters,
                                    conv_shortcut=False,
                                    name=f'{name}_blockv1_{filters}_{i}')
        return x

    def residual_block(self,
                       x,
                       filters=192,
                       kernel=3,
                       strides=(1, 1),
                       conv_shortcut=True,
                       activation='relu',
                       padding='same',
                       data_format='channels_last',
                       name='residual_block'):
        """A residual block"""

        bn_axis = 3 if data_format == 'channels_last' else 1
        # 4x12x13
        if conv_shortcut:
            shortcut = tf.keras.layers.Conv2D(
                filters,
                1,
                strides=strides,
                name=name + '_0_conv'
            )(x)
            shortcut = tf.keras.layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn'
            )(shortcut)
        else:
            shortcut = x
        # 4x12x192

        x = tf.keras.layers.Conv2D(
            filters,
            1,
            strides=strides,
            name=name + '_1_conv'
        )(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn'
        )(x)
        x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)

        # 4x12x192
        x = tf.keras.layers.Conv2D(
            filters,
            kernel,
            padding='SAME',
            name=name + '_2_conv'
        )(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn'
        )(x)
        x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)

        # 2x10x192
        x = tf.keras.layers.Conv2D(filters, 1, name=name + '_3_conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn'
        )(x)

        # 2x10x192
        x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
        x = tf.keras.layers.Activation('relu', name=name + '_out')(x)

        return x

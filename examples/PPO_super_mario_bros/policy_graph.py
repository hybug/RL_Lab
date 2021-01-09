'''
Author: hanyu
Date: 2021-01-06 10:13:41
LastEditTime: 2021-01-09 09:31:12
LastEditors: hanyu
Description: policy network of PPO
FilePath: /test_ppo/examples/PPO_super_mario_bros/policy_graph.py
'''
from ray_helper.miscellaneous import tf_model_ws


def warp_Model():
    '''
    description: warp the policy model
    param {*}
    return {Object: policy model}
    '''
    import tensorflow as tf
    from infer.categorical import categorical
    from utils.get_shape import get_shape

    @tf_model_ws
    class Model(object):
        def __init__(self,
                     act_space,
                     rnn,
                     use_rmc,
                     use_hrnn,
                     use_reward_prediction,
                     after_rnn,
                     use_pixel_control,
                     user_pixel_reconstruction,
                     scope='agent',
                     **kwargs):
            self.act_space = act_space
            self.use_rmc = use_rmc
            self.use_hrnn = use_hrnn
            self.scope = scope

            self.s_t = kwargs.get('s')
            self.prev_actions = kwargs.get('prev_a')
            self.prev_r = kwargs.get('prev_r')
            self.state_in = kwargs.get('state_in')

            prev_a = tf.one_hot(self.prev_actions,
                                depth=act_space, dtype=tf.float32)

            # Feature Network
            self.feature, self.cnn_feature, self.image_feature, self.state_out = self.feature_net(
                self.s_t, rnn, prev_a, self.prev_r, self.state_in, scope + '_current_feature')

            if use_hrnn:
                # TODO
                pass

            # Actor Network
            self.current_act_logits = self.a_net(
                self.feature, scope + '_acurrent')
            self.current_act = tf.squeeze(
                categorical(self.current_act_logits), axis=-1)

            # Critic Network
            self.current_value = self.v_net(self.feature, scope + '_vcurrent')

            advantage = kwargs.get('adv', None)
            if advantage is not None:
                # Adavantage Normalization
                # adv = (adv - adv_mean)
                # adv = adv / adv_std
                self.old_current_value = kwargs.get('v_cur')
                self.ret = advantage + self.old_current_value

                self.a_t = kwargs.get('a')
                self.behavior_logits = kwargs.get('a_logits')
                self.r_t = kwargs.get('r')

                self.adv_mean = tf.reduce_mean(advantage, axis=[0, 1])
                advantage -= self.adv_mean
                self.adv_std = tf.math.sqrt(
                    tf.reduce_mean(advantage ** 2, axis=[0, 1]))
                self.advantage = advantage / tf.maximum(self.adv_std, 1e-12)

                self.slots = tf.cast(kwargs.get('slots'), tf.float32)

                if use_reward_prediction:
                    # TODO
                    # reward prediction network
                    pass

                if user_pixel_reconstruction:
                    # TODO
                    # pixerl reconstruction network
                    pass

                if use_pixel_control:
                    # TODO
                    # pixel control network
                    pass

        def get_current_act(self):
            return self.current_act

        def get_current_logits(self):
            return self.current_act_logits

        def feature_net(self, image, rnn, prev_a, prev_r, state_in, scope='feature'):
            '''
            description: feature-extraction network
            param {
                image: the input image
                rnn: rnn network
                prev_a: previous action
                pre_v: previous value
                state_in: state_in using in rnn
            }
            return {
                Tensor[feature]: the feature input of actor&critic
                Tensor[cnn_feature]: the cnn_feature input of reward prediction net
                Tensor[image_feature]: the image_feature input of coex adm
                Tensor[state_out]: the state_out after feature_net
            }
            '''
            shape = get_shape(image)
            with tf.variable_scope(scope, tf.AUTO_REUSE):
                image = tf.reshape(image, [-1] + shape[-3:])
                filter = [16, 32, 32]
                kernel = [(3, 3), (3, 3), (5, 3)]
                stride = [(1, 2), (1, 2), (2, 1)]

                for i in range(len(filter)):
                    image = tf.layers.conv2d(
                        image,
                        filters=filter[i],
                        kernel_size=kernel[i][0],
                        strides=stride[i][0],
                        padding='valid',
                        activation=None,
                        name=f'conv_{i}'
                    )
                    image = tf.layers.max_pooling2d(
                        image,
                        pool_size=kernel[i][1],
                        strides=stride[i][1],
                        padding='valid',
                        name=f'max_pool_{i}'
                    )
                    image = self.residual_block(image, f'res0_{i}')
                image = tf.nn.relu(image)

                new_shape = get_shape(image)
                # the batch_size & seqlen dimensions remain the same
                image_feature = tf.reshape(
                    image, [shape[0], shape[1], new_shape[1], new_shape[2], new_shape[3]])

                feature = tf.reshape(
                    image, [shape[0], shape[1], new_shape[1] * new_shape[2] * new_shape[3]])

                cnn_feature = tf.layers.dense(
                    feature, 256, tf.nn.relu, name='feature')
                feature = tf.concat(
                    [cnn_feature, prev_a, prev_r[:, :, None]], axis=-1)

                if self.use_hrnn:
                    # TODO
                    pass
                elif self.use_rmc:
                    # TODO
                    pass
                else:
                    initial_state = tf.split(state_in, 2, axis=-1)
                    feature, c_out, h_out = rnn(
                        feature, initial_state=initial_state)
                    state_out = tf.concat([c_out, h_out], axis=-1)

            return feature, cnn_feature, image_feature, state_out

        def a_net(self, feature, scope):
            '''
            description: actor network
            param {feature: the output of feature_net}
            return {Tensor: the act_logits tensor}
            '''
            net = feature
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                net = tf.layers.dense(net, get_shape(
                    feature)[-1], activation=tf.nn.relu, name='dense')
                act_logits = tf.layers.dense(
                    net, self.act_space, activation=None, name='a_logits')
            return act_logits

        def v_net(self, feature, scope):
            '''
            description: value network as critic
            param {feature: the output of feature_net}
            return {Tensor: the v_value tensor}
            '''
            net = feature
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                net = tf.layers.dense(
                    net,
                    get_shape(feature)[-1],
                    activation=tf.nn.relu,
                    name='dense'
                )
                v_value = tf.squeeze(
                    tf.layers.dense(
                        net,
                        1,
                        activation=None,
                        name='v_value'
                    ),
                    axis=-1
                )
            return v_value

        def reconstruct_net(self):
            # TODO
            pass

        def control_net(self):
            # TODO
            pass

        @staticmethod
        def residual_block(input, scope):
            shape = get_shape(input)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                last_output = tf.nn.relu(input)
                last_output = tf.layers.conv2d(
                    last_output,
                    filters=shape[-1],
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    activation=None,
                    name='conv0'
                )
                last_output = tf.nn.relu(last_output)
                last_output = tf.layers.conv2d(
                    last_output,
                    filters=shape[-1],
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    activation=None,
                    name='conv1'
                )
                output = last_output + input
            return output

    return Model

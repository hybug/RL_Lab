'''
Author: hanyu
Date: 2021-06-24 06:46:05
LastEditTime: 2021-06-24 11:51:45
LastEditors: hanyu
Description: miscellaneous 
FilePath: /test_ppo/ray_helper/miscellaneous.py
'''


def tf_model_ws(cls):
    """the decorator using for setting & getting weights of tf graph
    """
    import re

    class ModelWeightsHelper(cls):
        def __init__(self, *args, **kwargs) -> None:
            super(ModelWeightsHelper, self).__init__(*args, **kwargs)
            self.set_ws_ops = None
            self.ws_ph_dict = None

        @staticmethod
        def get_ws(sess) -> dict:
            """get tensorflow graph's weights

            Args:
                sess (Session): tf.Session

            Returns:
                dict: the weights dict
            """
            import tensorflow as tf
            # get the graph's weights' tensor which trainable=True
            trainable_vars = tf.trainable_variables()

            # can we find the other way to get this session?
            ws = sess.run(trainable_vars)
            names = [re.match("^(.*):\\d+$", var.name).group(1)
                     for var in trainable_vars]
            return dict(zip(names, ws))

        def set_ws(self, sess, ws):
            """set the weighs into the graph

            Args:
                sess (Session): tf Session
                ws (dict): weights dict, {name: value}
            """
            if self.set_ws_ops is None:
                # there isn't any op to set weights, get the ops
                self.set_ws_ops = self._set_ws(ws)
            feed_dict = dict()
            for name, placeholder in self.ws_ph_dict.items():
                assert name in ws
                feed_dict[placeholder] = ws[name]
            sess.run(self.set_ws_ops, feed_dict=feed_dict)

        def _set_ws(self, to_ws):
            """complete the ws_ph_dict and return the set_ws ops

            Args:
                to_ws ([type]): weights
            """
            import tensorflow as tf
            trainable_vars = tf.trainable_variables()
            print(f'trainable vars: {trainable_vars}')
            names = [re.match("^(.*):\\d+$", var.name).group(1)
                     for var in trainable_vars]

            ops = list()
            names_to_trainable_vars = dict(zip(names, trainable_vars))

            placeholders_dict = dict()
            for name, var in names_to_trainable_vars.items():
                assert name in to_ws
                placeholder = tf.placeholder(
                    dtype=to_ws[name].dtype, shape=to_ws[name].shape)
                placeholders_dict[name] = placeholder
                op = tf.assign(var, placeholder)
                ops.append(op)
            self.ws_ph_dict = placeholders_dict
            return tf.group(ops)

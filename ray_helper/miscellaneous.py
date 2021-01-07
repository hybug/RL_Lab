'''
Author: hanyu
Date: 2021-01-07 10:07:40
LastEditTime: 2021-01-07 12:22:17
LastEditors: hanyu
Description: miscellaneous function
FilePath: /test_ppo/ray_helper/miscellaneous.py
'''
import os
import ray


def tf_model_ws(cls):
    '''
    description: set and load weights to tf graph
    param {cls}
    return {object: TF_Model_WS_Helper}
    '''
    import re

    class TF_Model_WS_Helper(cls):
        def __init__(self, *args, **kwargs):
            super(TF_Model_WS_Helper, self).__init__(*args, **kwargs)
            self.set_ws_ops = None
            self.ws_placeholders_dict = None

        @staticmethod
        def get_ws(sess):
            '''
            description: get weights from tf graph
            param {sess}
            return {Dict(name, weights): the weights}
            '''
            import tensorflow as tf
            trainable_vars = tf.trainable_variables()

            # sess = tf.get_default_session()
            # this function cant gat the correct sess
            # so set the sess as pramas

            ws = sess(trainable_vars)
            name_list = [re.match("^(.*):\\d+$", var.name).group(1)
                         for var in trainable_vars]
            return dict(zip(name_list, ws))

        def _set_ws(self, to_ws):
            '''
            description: set the weight to placeholder
            param {to_ws: name to var}
            return {Operation: op of a group of ops}
            '''
            import tensorflow as tf
            trainable_vars = tf.trainable_variables()
            print(f'trainable vars: {trainable_vars}')
            name_list = [re.match(
                "^(.*):\\d+$", var.name).group(1) for var in trainable_vars]

            ops = []
            name_list_2_trainable_vars = dict(zip(name_list, trainable_vars))
            placeholders_dict = dict()
            for name, var in name_list_2_trainable_vars.items():
                assert name in to_ws
                placeholder = tf.placeholder(
                    dtype=to_ws[name].dtype, shape=to_ws[name].shape)
                placeholders_dict[name] = placeholder
                op = tf.assign(var, placeholder)
                ops.append(op)
            self.ws_placeholders_dict = placeholders_dict
            return tf.group(ops)

        def set_ws(self, sess, ws):
            '''
            description: set the weights
            param {
                sess: tf session
                ws: weights
            }
            return {*}
            '''
            if self.set_ws_ops is None:
                self.set_ws_ops = self._set_ws
            feed_dict = dict()
            for key, value in self.ws_placeholders_dict.items():
                assert key in ws
                feed_dict[key] = ws[key]
            sess.run(self.set_ws_ops, feed_dict=feed_dict)

    return TF_Model_WS_Helper


def init_cluster_ray(log_to_driver=True):
    """
    connect to a exist ray cluster, if not exist init one
    :return:
    """
    server_hosts = os.getenv('ARNOLD_SERVER_HOSTS', None)
    assert server_hosts is not None
    server_ip, _ = server_hosts.split(',')[0].split(':')
    redis_port = int(os.environ['ARNOLD_RUN_ID']) % 1e4 + 6379
    ray.init(address=':'.join(
        [server_ip, str(int(redis_port))]), log_to_driver=log_to_driver)


"""
    hdfs helper function
"""


def warp_exists(path, use_hdfs=False):
    if use_hdfs:
        import pyarrow as pa
        fs = pa.hdfs.connect()
        return fs.exists(path)
    else:
        return os.path.exists(path)


def warp_mkdir(dir_name, use_hdfs=False):
    if use_hdfs:
        import pyarrow as pa
        fs = pa.hdfs.connect()
        fs.mkdir(dir_name)
    else:
        os.mkdir(dir_name)

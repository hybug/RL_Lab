import os
import ray


def init_cluster_ray(log_to_driver=True):
    """
    connect to a exist ray cluster, if not exist init one
    :return:
    """
    server_hosts = os.getenv('ARNOLD_SERVER_HOSTS', None)
    assert server_hosts is not None
    server_ip, _ = server_hosts.split(',')[0].split(':')
    redis_port = int(os.environ['ARNOLD_RUN_ID']) % 1e4 + 6379
    ray.init(address=':'.join([server_ip, str(int(redis_port))]), log_to_driver=log_to_driver)


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
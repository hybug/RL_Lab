# coding: utf-8

from algorithm import dPPOcC
from module import KL_from_gaussians
from module import mse
from module import coex
from module import icm
from module import TmpHierRNN
from module import TmpHierRMCRNN
from module import entropy_from_logits as entropy
from train_ops import miniOp
from module import RMCRNN
from utils import get_shape
import tensorflow as tf
import time
import os
import logging
from ray_helper.miscellaneous import init_cluster_ray, warp_mkdir, warp_exists
from ray_helper.rollout_collector import QueueReader, fetch_one_structure
from ray_helper.asyncps import AsyncPS
from ray_helper.rollout_collector import RolloutCollector
import ray
import sys

sys.path.append("/opt/tiger/test_ppo")


logging.getLogger('tensorflow').setLevel(logging.ERROR)

NEST = tf.contrib.framework.nest

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train", "mode")
flags.DEFINE_integer("act_space", 12, "act space")

flags.DEFINE_string(
    "basedir_ceph", "/mnt/cephfs_new_wj/arnold/labcv/xiaochangnan"
                    "/PPOcGAE_SuperMarioBros-v0", "base dir for ceph")
flags.DEFINE_string(
    "basedir_hdfs",
    'hdfs://haruna/home/byte_arnold_lq_mlsys/user/xiaochangnan/test_ppo',
    "base dir for hdfs")
flags.DEFINE_string("dir", "0", "dir number")
flags.DEFINE_string(
    "scriptdir",
    "/opt/tiger/test_ppo/examples/PPO_super_mario_bros",
    "script dir")

flags.DEFINE_bool("use_stage", True, "whether to use tf.contrib.staging")

flags.DEFINE_integer("use_rmc", 0, "whether to use rmcrnn instead of lstm")
flags.DEFINE_integer("use_hrmc", 1, "whether to use tmp hierarchy rmcrnn")
flags.DEFINE_integer(
    "use_hrnn", 0, "whether to use tmp hierarchy rnn (lstm+rmc)")
flags.DEFINE_bool("use_icm", False, "whether to use icm during training")
flags.DEFINE_bool("use_coex", False, "whether to use coex adm during training")
flags.DEFINE_bool("use_reward_prediction", True,
                  "whether to use reward prediction")
flags.DEFINE_integer(
    "after_rnn", 1, "whether to use reward prediction after rnn")
flags.DEFINE_integer("use_pixel_control", 1, "whether to use pixel control")
flags.DEFINE_integer("use_pixel_reconstruction", 0,
                     "whether to use pixel reconstruction")

flags.DEFINE_float("pq_kl_coef", 0.1,
                   "weight of kl between posterior and prior")
flags.DEFINE_float("p_kl_coef", 0.01,
                   "weight of kl between prior and normal gaussian")

flags.DEFINE_bool("use_hdfs", True, "whether to use hdfs")

flags.DEFINE_integer("parallel", 64, "parallel envs")
flags.DEFINE_integer("max_steps", 3200, "max rollout steps")
flags.DEFINE_integer("seqlen", 32, "seqlen of each training segment")
flags.DEFINE_integer("burn_in", 32, "seqlen of each burn-in segment")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_integer("total_environment_frames", 1000000000,
                     "total num of frames for train")

flags.DEFINE_float("init_lr", 1e-3, "initial learning rate")
flags.DEFINE_float("lr_decay", 1.0, "whether to decay learning rate")
flags.DEFINE_float("warmup_steps", 4000, "steps for warmup")

flags.DEFINE_integer("frames", 1, "stack of frames for each state")
flags.DEFINE_integer("image_size", 84, "input image size")
flags.DEFINE_float("vf_clip", 1.0, "clip of value function")
flags.DEFINE_float("ppo_clip", 0.2, "clip of ppo loss")
flags.DEFINE_float("gamma", 0.99, "discount rate")
flags.DEFINE_float("pi_coef", 10.0, "weight of policy fn loss")
flags.DEFINE_float("vf_coef", 1.0, "weight of value fn loss")
flags.DEFINE_float("ent_coef", 1.0, "weight of entropy loss")
flags.DEFINE_bool("zero_init", False, "whether to zero init initial state")
flags.DEFINE_float("grad_clip", 40.0, "global grad clip")

flags.DEFINE_integer("seed", 12358, "random seed")

# param for ray evaluator
flags.DEFINE_float("timeout", 0.1, "get operation timeout")
flags.DEFINE_integer("num_returns", 32, "nof data of wait operation")
flags.DEFINE_integer('cpu_per_actor', 2,
                     'nof cpu required for infserver, -1 for not require')
flags.DEFINE_integer('load_ckpt_period', 10,
                     'for how many step to load ckpt in inf server')
flags.DEFINE_integer(
    'qsize', 8, 'for how many qsize * batchsize in main procress')
flags.DEFINE_integer('nof_server_gpus', 1, 'nof_gpus for training')
flags.DEFINE_integer('nof_evaluator', 1, 'nof_gpus for training')


def build_policy_evaluator(kwargs):
    """
    construct policy_evaluator
    :params: kwargs
    :return: construcation method and params of Evaluator
    """
    from copy import deepcopy

    if kwargs["use_hrnn"]:
        kwargs["state_size"] = 1 + (8 + 2 + 8) * 4 * 64
    elif kwargs["use_hrmc"]:
        kwargs["state_size"] = 1 + (8 + 4 + 4) * 4 * 64
    elif kwargs["use_rmc"]:
        kwargs['state_size'] = 64 * 4 * 4
    else:
        kwargs['state_size'] = 256 * 2

     env_kwargs = deepcopy(kwargs)
    env_kwargs['action_repeats'] = [1]

    model_kwargs = deepcopy(kwargs)
    # pickle func in func
    from examples.PPO_super_mario_bros.env import build_env
    from examples.PPO_super_mario_bros.policy_graph import build_evaluator_model
    return model_kwargs, build_evaluator_model, env_kwargs, build_env

def init_dir_and_log():
    tf.set_random_seed(FLAGS.seed)
    if FLAGS.use_hdfs:
        base_dir = FLAGS.basedir_hdfs
    else:
        base_dir = FLAGS.basedir_ceph
    base_dir = os.path.join(base_dir, FLAGS.dir)

    if not warp_exists(base_dir, use_hdfs=FLAGS.use_hdfs):
        warp_mkdir(base_dir, FLAGS.use_hdfs)

    ckpt_dir = os.path.join(base_dir, "ckpt")
    if not warp_exists(ckpt_dir, FLAGS.use_hdfs):
        warp_mkdir(ckpt_dir, FLAGS.use_hdfs)

    local_log_dir = os.path.join(FLAGS.scriptdir, 'log')
    if not os.path.exists(local_log_dir):
        os.mkdir(local_log_dir)
    logging.basicConfig(filename=os.path.join(
        local_log_dir, "Trainerlog"), level="INFO")

    summary_dir = os.path.join(base_dir, "summary")
    if not warp_exists(summary_dir, FLAGS.use_hdfs):
        warp_mkdir(summary_dir, FLAGS.use_hdfs)
    return base_dir, ckpt_dir, summary_dir

def train():
    """
    init dir and log config
    """
    init_cluster_ray()
    base_dir, ckpt_dir, summary_dir = init_dir_and_log()

    kwargs = FLAGS.flag_values_dict()
    kwargs["BASE_DIR"] = base_dir
    kwargs["ckpt_dir"] = ckpt_dir
    """
    get one seg from rollout worker for dtype and shapes

    :param kwargs rollout worker config
    """
    logging.info('get one seg from Evaluator for dtype and shapes')
    ps = AsyncPS.remote()
    small_data_collector = RolloutCollector(
        server_nums=1, ps=ps, policy_evaluator_build_func=build_policy_evaluator,
        **kwargs)
    cache_struct_path = f'/tmp/{FLAGS.dir}.pkl'
    structure = fetch_one_structure(small_data_collector, cache_struct_path=cache_struct_path, is_head=True)
"""
Author: hanyu
Date: 2020-12-29 14:34:34
LastEditTime: 2020-12-29 14:39:09
LastEditors: hanyu
Description: PPO Trainer with ray core implement
FilePath: /test_ppo/examples/PPO_super_mario_bros/ray_trainer.py
"""

import logging
import os
import sys
import time

import ray
import tensorflow as tf
from algorithm.dPPOcC import dPPOcC
from module import RMCRNN, KL_from_gaussians, TmpHierRMCRNN, TmpHierRNN, coex
from module import entropy_from_logits as entropy
from module import icm, mse
from ray_helper.asyncps import AsyncPS
from ray_helper.miscellaneous import init_cluster_ray, warp_exists, warp_mkdir
from ray_helper.rollout_collector import (QueueReader, RolloutCollector,
                                          fetch_one_structure)
from train_ops import miniOp
from utils import get_shape

sys.path.append("/opt/tiger/test_ppo")


logging.getLogger('tensorflow').setLevel(logging.ERROR)

NEST = tf.contrib.framework.nest

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train", "mode")
flags.DEFINE_integer("act_space", 12, "act space")

flags.DEFINE_string(
    "basedir_ceph", "/notebooks/projects/hanyu/RayProject/test_ppo/temp_base_file"
                    "/PPOcGAE_SuperMarioBros-v0", "base dir for ceph")
flags.DEFINE_string(
    "basedir_hdfs",
    '',
    "base dir for hdfs")
flags.DEFINE_string("dir", "0", "dir number")
flags.DEFINE_string(
    "scriptdir",
    "/notebooks/projects/hanyu/RayProject/test_ppo/examples/PPO_super_mario_bros",
    "script dir")

flags.DEFINE_bool("use_stage", True, "whether to use tf.contrib.staging")

flags.DEFINE_integer(
    "use_rmc", 0, "whether to use rmcrnn(Relational Memroy Core RNN) instead of lstm")
flags.DEFINE_integer("use_hrmc", 1, "whether to use tmp hierarchy rmcrnn")
flags.DEFINE_integer(
    "use_hrnn", 0, "whether to use tmp hierarchy rnn (lstm+rmc)")
flags.DEFINE_bool("use_icm", False,
                  "whether to use icm(Intrinsic Curiosity Module) during training")
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

flags.DEFINE_bool("use_hdfs", False, "whether to use hdfs")

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
flags.DEFINE_float("entropy_coeff", 1.0, "weight of entropy loss")
flags.DEFINE_bool("zero_init", False, "whether to zero init initial state")
flags.DEFINE_float("grad_clip", 40.0, "global grad clip")

flags.DEFINE_integer("seed", 12358, "random seed")

# param for ray evaluator
flags.DEFINE_float("timeout", 0.1, "get operation timeout")
flags.DEFINE_integer("num_returns", 1, "number of data of wait operation")
flags.DEFINE_integer('cpu_per_actor', 2,
                     'number of cpu required for infserver, -1 for not require')
flags.DEFINE_integer('load_ckpt_period', 10,
                     'for how many step to load ckpt in inf server')
flags.DEFINE_integer(
    'qsize', 8, 'for how many qsize * batchsize in main procress')
flags.DEFINE_integer('nof_server_gpus', 1, 'number of gpus for training')
flags.DEFINE_integer('nof_evaluator', 1, 'number of cpus for training')

Model = warp_Model()


def build_learner(predatas, postdatas, act_space, num_frames):
    '''
    description: builder the learner
    param {*}
    return {
        Tensor: number of frames
        Tensor: number of global step
    }
    '''
    global_step = tf.train.get_or_create_global_step()
    init_lr = FLAGS.init_lr
    decay = FLAGS.lr_decay
    warmup_steps = FLAGS.warmup_steps
    use_rmc = FLAGS.use_rmc
    use_hrmc = FLAGS.use_hrmc
    use_hrnn = FLAGS.use_hrnn
    use_icm = FLAGS.use_icm
    use_coex = FLAGS.use_coex
    use_reward_prediction = FLAGS.use_reward_prediction
    after_rnn = FLAGS.after_rnn
    use_pixel_control = FLAGS.use_pixel_control
    use_pixel_reconstruction = FLAGS.use_pixel_reconstruction
    pq_kl_coef = FLAGS.pq_kl_coef
    pq_kl_coef = FLAGS.p_kl_coef

    global_step_float = tf.cast(global_step, tf.float32)

    lr = tf.train.polynomial_decay(
        init_lr,
        global_step,
        FLAGS.total_environment_frames // (FLAGS.batch_size * FLAGS.seqlen),
        init_lr / 10.
    )
    is_warmup = tf.cast(global_step_float < warmup_steps, tf.float32)
    lr = is_warmup * global_step_float / warmup_steps * init_lr + \
        (1.0 - is_warmup) * (init_lr * (1.0 - decay) + lr * decay)
    optimizer = tf.train.AdamOptimizer(lr)

    entropy_coeff = tf.train.polynomial_decay(
        FLAGS.entropy_coeff,
        global_step,
        FLAGS.total_environment_frames // (FLAGS.batch_size * FLAGS.seqlen),
        FLAGS.entropy_coeff / 10.
    )

    if FLAGS.zero_init:
        predatas["state_in"] = tf.zeros_like(predatas["state_in"])

    if use_hrnn:
        # TODO
        pass
    elif use_hrmc:
        # TODO
        pass
    elif use_rmc:
        # TODO
        pass
    else:
        rnn = tf.compat.v1.keras.layers.LSTM(
            256, return_sequence=True, return_state=True, name='lstm'
        )
    pre_model = Model(act_space, rnn, use_rmc, use_hrmc or use_hrnn, use_reward_prediction,
                      after_rnn, use_pixel_control, use_pixel_reconstruction, 'agent', **predatas)

    postdatas['state_in'] = tf.stop_gradient(pre_model.state_out)

    post_model = Model(act_space, rnn, use_rmc, use_hrmc or use_hrnn, use_reward_prediction,
                       after_rnn, use_pixel_control, use_pixel_reconstruction, 'agent', **postdatas)

    tf.summary.scalar('adv_mean', post_model.adv_mean)
    tf.summary.scalar('adv_std', post_model.adv_std)

    losses = dPPOcC(action=post_model.a_t,
                    policy_logits=post_model.current_act_logits,
                    behavior_logits=post_model.behavior_logits,
                    advantage=post_model.advantage,
                    policy_clip=FLAGS.ppo_clip,
                    vf=post_model.current_value,
                    vf_target=post_model.ret,
                    value_clip=FLAGS.vf_clip,
                    old_vf=post_model.old_current_value)


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
    from examples.PPO_super_mario_bros.policy_graph import \
        build_evaluator_model
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
    kwargs["num_returns"] = FLAGS.num_returns
    """
    get one seg from rollout worker for dtype and shapes

    :param kwargs rollout worker config
    """
    logging.info('get one seg from Evaluator for dtype and shapes')
    ps = AsyncPS.remote()
    # get one seg for dtype and shapes, so the server_nums=1
    small_data_collector = RolloutCollector(
        server_nums=1, ps=ps, policy_evaluator_build_func=build_policy_evaluator,
        **kwargs)
    cache_struct_path = f'/tmp/{FLAGS.dir}.pkl'
    structure = fetch_one_structure(small_data_collector, cache_struct_path=cache_struct_path, is_head=True)
    # after get example seg, del this small_data_collector
    del small_data_collector

    """
    init data prefetch thread, prepare_input_pipe
    """
    # keys = ['s', 'a', 'prev_a', 'a_logits', 'r', 'prev_r', 'adv', 'v_cur', 'state_in', 'slots']
    keys = list(structure.keys())
    dtypes = [structure[k].dtype for k in keys]
    shapes = [structure[k].shape for k in keys]
    
    segBuffer = tf.queue.FIFOQueue(
        capacity=FLAGS.qsize * FLAGS.batch_size,
        dtypes=dtypes,
        shapes=shapes,
        names=keys,
        shared_name='buffer'
    )

    server_nums = FLAGS.nof_evaluator
    server_nums_refine = server_nums * 2 // FLAGS.cpu_per_actor
    nof_server_gpus = FLAGS.nof_server_gpus
    server_nums_refine = server_nums_refine // nof_server_gpus
    # instantiate the RolloutCollector
    data_collector = RolloutCollector(server_nums=server_nums_refine, ps=ps,
                                    policy_evaluator_build_func=build_policy_evaluator, **kwargs)

    config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOption(per_process_gpu_memory_fraction=1)
    )
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    # Start a Thread to implement QueueReader
    # get data from data_collector, then enqueue into segBuffer continuously 
    reader = QueueReader(
        sess=sess,
        global_queue=segBuffer,
        data_collector=data_collector,
        keys=keys,
        dtypes=dtypes,
        shapes=shapes
    )
    reader.daemon = True
    reader.start()

    # s_t: (batch_size, seqlen, 84, 84, frames)
    # a: (batch_size, seqlen)
    # prev_a: (batch_size, seqlen)
    # a_logits: (batch_size, seqlen, act_space)
    # r: (batch_size, seqlen)
    # prev_r: (batch_size, seqlen)
    # adv: (batch_size, seqlen)
    # v_cur: (batch_size, seqlen)
    # state_in: (batch_size, state_size)
    # slots: (batch_size, seqlen)
    dequeued = segBuffer.dequeue_many(FLAGS.batch_size)
    pre_placeholders, post_placeholders = dict(), dict()
    for k, v in dequeued.items():
        if k == 'state_in':
            pre_placeholders = v
        else:
            # notes TODO
            pre_placeholders[k], post_placeholders = tf.split(
                v, [FLAGS.burn_in, FLAGS.seqlen], axis=1
            )
    prekeys = list(pre_placeholders.keys())
    postkeys = list(post_placeholders.keys())

    # count frame&total steps and summary into tensorboard
    num_frames = tf.get_variable(
        'num_environment_frames',
        initializer=tf.zeros_initializer(),
        shape=[],
        dtype=tf.int32,
        trainable=False
    )
    tf.summary.scalar("num_frames", num_frames)
    global_step = tf.train.get_or_create_global_step()

    dur_time_tensor = tf.placeholder(dtype=tf.float32)
    tf.summary.scalar("time_per_step", dur_time_tensor)

    # set stage_op and build learner
    with tf.device('/gpu'):
        if FLAGS.use_stage:
            # TODO
            pass
        else:
            stage_op = []
            pre_datas, post_datas = pre_placeholders, post_placeholders

        act_space = FLAGS.act_space
        num_frames_and_train, global_step_and_train = build_learner(

        )

def main():
    if FLAGS.mode == 'train':
        train()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

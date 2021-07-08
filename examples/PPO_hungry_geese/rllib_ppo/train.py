'''
Author: hanyu
Date: 2021-06-16 12:01:42
LastEditTime: 2021-07-08 08:22:08
LastEditors: hanyu
Description: train
FilePath: /RL_Lab/examples/PPO_hungry_geese/rllib_ppo/train.py
'''
import argparse
import random
import sys

import ray
from examples.PPO_hungry_geese.rllib_ppo.env import warp_env
from examples.PPO_hungry_geese.rllib_ppo.MaskedResNetClass import \
    MaskedResidualNetwork
from ray import tune
from ray.rllib import agents
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env


def env_creator(env_config):
    HungryGeeseEnv = warp_env()
    return HungryGeeseEnv()


def get_trainer_config(args):
    if args.model_type == "MaskedResNet":
        model_config = {
            'custom_model': 'MaskedResNet',
            'custom_model_config': {'conv_activation': 'relu',
                                    'conv_filters': [(64, 1, 1, 4), (128, 1, 1, 3), (256, 1, 1, 2), (128, (7, 11), 1, 1)],
                                    'no_final_linear': False,
                                    'vf_share_layers': False,
                                    'rows': 7,
                                    'columns': 11,
                                    'channels': 17}
        }
    else:
        model_config = 'MODEL_DEFAULTS'

    def policy_mapping_fn(agent_id):
        pol_id = random.choice(policy_ids)
        return pol_id

    def gen_policy(i):
        env = env_creator({})
        config = {
            "model": {
                "custom_model": ["MaskedResNet1", "MaskedResNet2"][i % 2],
            },
            "gamma": random.choice([0.95, 0.99]),
        }
        return (None, env.observation_space, env.action_space, config)

    policies = {
        f'policy_{i}': gen_policy(i) for i in range(args.num_policies)
    }
    policy_ids = list(policies.keys())

    trainer_config = {
        'env': 'HungryGeeseEnv',
        'num_workers': args.ray_workers,
        'num_envs_per_worker': 1,
        'gamma': args.gamma,  # 0.99
        'lr': 1e-5,  # 3e-4 1e-5
        # 'lr_schedule': [
        #     [0, 3e-4],
        #     [3e6, 1e-5]
        # ],
        'vf_share_layers': False,
        "vf_loss_coeff": 0.5,
        'train_batch_size': args.train_batch_size,
        # 'clip_param':0.3, # PPOclip系数
        "rollout_fragment_length": args.rollout_fragment_length,  # 200
        "sgd_minibatch_size": 512,
        "num_sgd_iter": args.num_sgd_iter,
        'kl_coeff': 0.2,
        'kl_target': 0.01,
        'clip_rewards': False,  # True
        # 'entropy_coeff': 0.001,
        'entropy_coeff_schedule': [
            [0, 0.01],
            [5e7, 0]
        ],

        'model': model_config,

        'num_gpus': args.num_gpus,
        'explore': True,

        'multiagent': {
            'policies': policies,
            'policy_mapping_fn': policy_mapping_fn
        }
    }
    return trainer_config


def main(args):
    ray.init(address=args.ray_address)

    register_env("HungryGeeseEnv", env_creator)

    if args.model_type == "MaskedResNet":
        ModelCatalog.register_custom_model(
            "MaskedResNet1", MaskedResidualNetwork)
        ModelCatalog.register_custom_model(
            "MaskedResNet2", MaskedResidualNetwork)
    else:
        sys.exit()

    trainer_config = get_trainer_config(args)
    stop_config = {'training_iteration': 10000}

    is_training = True
    while is_training:
        tune.run(agents.ppo.PPOTrainer, config=trainer_config, stop=stop_config, name=args.exper_name,
                 local_dir=args.model_dir,
                 restore=args.model_restore,
                 checkpoint_freq=args.checkpoint_freq, checkpoint_at_end=True,
                 )


def parse_argument(argv):
    parser = argparse.ArgumentParser(description='SRCNN')
    parser.add_argument('--model_type', type=str,
                        default="MaskedResNet", help='模型类型')
    parser.add_argument('--model_dir', type=str,
                        default="result/hungry_geese/", help='模型保存位置')
    parser.add_argument('--model_restore', type=str,
                        default=None, help='restore模型位置参数')
    parser.add_argument('--exper_name', type=str,
                        default="test", help='试验名称')
    parser.add_argument('--ray_workers', type=int,
                        default=1, help='Ray启动Worker数量')
    parser.add_argument('--rollout_fragment_length', type=int,
                        default=512, help='每个worker采样的样本数')
    parser.add_argument('--gamma', type=float,
                        default=0.8, help='折扣因子')
    parser.add_argument('--num_gpus', type=int, default=0, help='GPU显卡数量')
    parser.add_argument('--train_batch_size', type=int,
                        default=512, help='训练batch size')
    parser.add_argument('--checkpoint_freq', type=int,
                        default=100, help='保存checkpoint频率，多少次迭代后保存一次')
    parser.add_argument('--num_sgd_iter', type=int,
                        default=1, help='minibatch reuse的次数')
    parser.add_argument('--num_policies', type=int,
                        default=4, help='policy的数量')
    parser.add_argument('--ray_address', type=str,
                        default=None, help='ray address')

    args = parser.parse_args(argv)

    return args


if __name__ == "__main__":
    main(parse_argument(sys.argv[1:]))

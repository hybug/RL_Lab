'''
Author: hanyu
Date: 2022-07-19 11:32:24
LastEditTime: 2022-08-03 11:37:53
LastEditors: hanyu
Description: basic config
FilePath: /RL_Lab/configs/config_base.py
'''
from dataclasses import dataclass, field

import yaml


def singleton(class_):
    instances = dict()

    def get_instances(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return get_instances


@dataclass
class ParamsBase:
    config_file: str = str()

    def __post_init__(self):
        from get_logger import BASEDIR, logger

        if self.config_file:
            logger.info(f"Loading config from {self.config_file}")

            try:
                with open(f"{BASEDIR}/{self.config_file}") as f:
                    data = yaml.safe_load(f)
                logger.info(f"Loaded parameters from {self.config_file}")
                return True, data
            except Exception as e:
                logger.error(f"Error >> {str(e)}")
                return False, None
        else:
            logger.error("No config file specified")
            return False, None


@dataclass
class EnvParams:
    # Basic Params
    num_envs: int = 1
    env_name: str = ""
    experiment_name: str = ""
    reward: str = ""
    seed: int = 0
    frame_stacking: bool = False
    frames_stack_size: int = 3


@dataclass
class TrainParams:

    trainer: str = ""
    # Choose from architecture defined in network.py
    nn_architecure: str = ''

    # Training Hyperparameters
    epochs: int = 1000  # Number of epochs
    steps_per_epoch: int = 250  # Steps per epoch
    num_mini_batches: int = 4
    gamma: float = 0.99
    lam: float = 0.97
    seed: int = EnvParams.seed
    save_freq: int = 5

    restore_path: str = None


@dataclass
class PolicyParams:

    # Alogrithm Hyperparameters
    lr: float = 0.001
    train_iters: int = 5

    clip_ratio: float = 0.2
    target_kl: float = 0.01
    ent_coef: float = 0.1
    v_coef: float = 0.5
    clip_grads: float = 0.5

    # Network Hyperparameters
    cnn_filters: list = field(default_factory=list)
    pool_size: list = field(default_factory=list)
    mlp_filters: list = field(default_factory=list)
    activation: str = "relu"
    output_activation: str = "tanh"
    kernel_initializer: str = "glorot_uniform"
    input_shape: tuple = field(default_factory=tuple)
    act_size: int = field(default_factory=int)
    output_size: int = 19


@singleton
@dataclass
class Params(ParamsBase):
    env: EnvParams = None
    trainer: TrainParams = None
    policy: PolicyParams = None

    def __post_init__(self):
        _has_config_file, data = super().__post_init__()

        if _has_config_file:
            self.env = EnvParams(**data['env_params'])
            self.trainer = TrainParams(**data['train_params'])
            self.policy = PolicyParams(**data['policy_params'])

'''
Author: hanyu
Date: 2022-07-19 11:46:16
LastEditTime: 2022-08-26 15:33:35
LastEditors: hanyu
Description: ppo config
FilePath: /RL_Lab/alogrithm/ppo/ppo_config.py
'''

from dataclasses import dataclass, field
from configs.config_base import ParamsBase, singleton


@dataclass
class PPOEnvParams:
    # Football Params
    env_name: str = field(repr=False)

    # Basic Params
    num_envs: int = 2
    experiment_name: str = ""
    seed: int = 0
    frame_stacking: bool = False
    frames_stack_size: int = 3


@dataclass
class PPOTrainParams:

    trainer: str = "Proximal Policy Optimization"

    # Training Hyperparameters
    epochs: int = 1000  # Number of epochs
    steps_per_epoch: int = 250  # Steps per epoch
    num_mini_batches: int = 4
    gamma: float = 0.99
    lam: float = 0.97
    seed: int = PPOEnvParams.seed
    save_freq: int = 5

    restore_path: str = None


@dataclass
class PPOPolicyParams:

    # Alogrithm Hyperparameters
    lr: float = 0.001
    train_iters: int = 5

    clip_param: float = 0.2
    vf_clip_param: float = 10
    target_kl: float = 0.01
    ent_coef: float = 0.1
    vf_loss_coef: float = 0.5
    kl_coef: float = 0.2
    clip_grads: float = 0.5

    # Network Hyperparameters
    # Choose from architecture defined in network.py
    nn_architecure: str = ''
    resnet_filters: list = field(default_factory=list)
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
class PPOParams(ParamsBase):
    env: PPOEnvParams = None
    trainer: PPOTrainParams = None
    policy: PPOPolicyParams = None

    def __post_init__(self):
        _has_config_file, data = super().__post_init__()

        if _has_config_file:
            self.env = PPOEnvParams(**data['env_params'])
            self.trainer = PPOTrainParams(**data['train_params'])
            self.policy = PPOPolicyParams(**data['policy_params'])

'''
Author: hanyu
Date: 2022-08-19 17:51:16
LastEditTime: 2022-08-23 20:38:11
LastEditors: hanyu
Description: 
FilePath: /RL_Lab/tests/test_rllib.py
'''
import ray
from ray import tune
# Import the RL algorithm (Trainer) we would like to use.
from ray.rllib.agents.ppo import PPOTrainer

ray.init(local_mode=True)
# ray.init()
# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "BeamRiderNoFrameskip-v4",
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 1,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "tf2",
    "lambda": 0.95,
    "kl_coeff": 0.5,
    "clip_rewards": True,
    "clip_param": 0.1,
    "vf_clip_param": 10,
    "entropy_coeff": 0.01,
    "train_batch_size": 5000,
    "rollout_fragment_length": 100,
    "sgd_minibatch_size": 500,
    "num_sgd_iter": 10,
    "batch_mode": "truncate_episodes"
}

# Create our RLlib Trainer.
# trainer = PPOTrainer(config=config)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
# for _ in range(10):
#     print(trainer.train())
stop_config = {"training_iteration": 10}
tune.run(
    PPOTrainer,
    config=config,
    stop=stop_config,
    local_dir="./tests/test_rllib/")

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
# trainer.evaluate()
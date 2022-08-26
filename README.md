<!--
 * @Author: hanyu
 * @Date: 2022-07-26 18:09:27
 * @LastEditTime: 2022-08-26 11:01:14
 * @LastEditors: hanyu
 * @Description: read me
 * @FilePath: /RL_Lab/README.md
-->
- [1. RL-Lab](#1-rl-lab)
- [Algorithms](#algorithms)
  - [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
- [Game Environment](#game-environment)
  - [Gym CartPole-v1](#gym-cartpole-v1)
    - [Experiment Result](#experiment-result)
  - [Gym Atari BeamRiderNoFrameskip-v4](#gym-atari-beamridernoframeskip-v4)
    - [Reproduce script](#reproduce-script)
    - [Experiment Result](#experiment-result-1)
  - [Google Football](#google-football)

# 1. RL-Lab
The implementation of some reinforcement learning algorithms refers to many open source projects on GitHub, and the framework design mainly refers to the implementation of rllib.
At present, the following algorithms and game environments have been implemented:

| Algorithms                               | Framework | Discrete Actions | Continuous Action |
| ---------------------------------------- | --------- | ---------------- | ----------------- |
| [PPO](#proximal-policy-optimization-ppo) | tf        | Yes              | Yes               |

| GameEnv                                                                 | Demo ALgorithms |
| ----------------------------------------------------------------------- | --------------- |
| [Gym CartPole-v1](#gym-cartpole-v1)                                     | PPO             |
| [Gym atari BeamRiderNoFrameskip-v4](#gym-atari-beamridernoframeskip-v4) | PPO             |
| [Gfootball](#google-football)                                           | PPO             |


# Algorithms
## Proximal Policy Optimization (PPO)

# Game Environment
## Gym CartPole-v1
[CardPole-v1 Reference](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)
### Experiment Result

## Gym Atari BeamRiderNoFrameskip-v4
[BeamRider Reference](https://github.com/openai/atari-py)
### Reproduce script
run `PYTHONPATH=./ python tests/test_ppotrainer_atari.py`
### Experiment Result


## Google Football


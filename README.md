<!--
 * @Author: hanyu
 * @Date: 2022-07-26 18:09:27
 * @LastEditTime: 2022-08-26 11:35:03
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
TODO

# Game Environment

## Gym CartPole-v1
[CardPole-v1 Reference](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)

### Experiment Result

## Gym Atari BeamRiderNoFrameskip-v4
[BeamRider Reference](https://github.com/openai/atari-py)

### Reproduce script
run `PYTHONPATH=./ python tests/test_ppotrainer_atari.py`

To reproduce the training result of deepmind/rllib, some of the following tricks are important:
1. DeepMind atari enviroment wrappers, see `envs/wrappers/atari_wrappers.py` copid from rllib in fact
    a. MonitorEnv for recording statistical variabels like _episode_reward * _episode_length

    b. NoopResetEnv for doing some No-Operation after environment reset
    
    c. MaxAndSkipEnv for Returning only every 4-th frame
    
    d. WarpFrame for warrper the observation to 84x84x1
    
    e. FireResetEnv for doing the fire-reseting
    
    f. EpisodicLifeEnv for making one episode for one life
    
    g. FrameStack for multi-frame-stack to observation
2. Set clip_reward True, clip the reward to [-1, 1]
3. Learning rate: 5e-5, epsilon: 1e-7
4. Other hparam according to config

### Experiment Result
![PPO_BreamRider](/docs/experiment_results/ppo/PPO_BeamRider.png "PPO_BreamRider")

Important reference indicators
1. Episode Reward: According to your reward engine, the bigger the better
2. explained_variance: It reflects the difference between your critic VF_OUT and VALUE_TARGET; In general, the closer to 1, the better

## Google Football
TODO


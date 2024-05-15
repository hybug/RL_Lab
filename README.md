<!--
 * @Author: hanyu
 * @Date: 2022-07-26 18:09:27
 * @LastEditTime: 2022-08-26 15:49:43
 * @LastEditors: hanyu
 * @Description: read me
 * @FilePath: /RL_Lab/README.md
-->
- [1. RL-Lab](#1-rl-lab)
- [2. Algorithms](#2-algorithms)
  - [2.1. Proximal Policy Optimization (PPO)](#21-proximal-policy-optimization-ppo)
- [3. Game Environment](#3-game-environment)
  - [3.1. Gym CartPole-v1](#31-gym-cartpole-v1)
    - [Reproduce script](#reproduce-script)
    - [3.1.1. Experiment Result](#311-experiment-result)
  - [3.2. Gym Atari BeamRiderNoFrameskip-v4](#32-gym-atari-beamridernoframeskip-v4)
    - [3.2.1. Reproduce script](#321-reproduce-script)
    - [3.2.2. Experiment Result](#322-experiment-result)
  - [3.3. Google Football](#33-google-football)

# 1. RL-Lab
The implementation of some reinforcement learning algorithms refers to many open source projects on GitHub, and the framework design mainly refers to the implementation of rllib.
At present, the following algorithms and game environments have been implemented:

| Algorithms                               | Framework | Discrete Actions | Continuous Action |
| ---------------------------------------- | --------- | ---------------- | ----------------- |
| [PPO](#proximal-policy-optimization-ppo) | tf        | Yes              | Yes               |

| GameEnv                                                                 | Demo ALgorithms |
| ----------------------------------------------------------------------- | --------------- |
| [Gym CartPole-v1](#gym-cartpole-v1)                                     | PPO             |
| [Gym Atari BeamRiderNoFrameskip-v4](#gym-atari-beamridernoframeskip-v4) | PPO             |
| [Gfootball](#google-football)                                           | PPO             |


# 2. Algorithms
## 2.1. Proximal Policy Optimization (PPO)
TODO

# 3. Game Environment

## 3.1. Gym CartPole-v1
[CardPole-v1 Reference](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)

### Reproduce script
run `PYTHONPATH=./ python tests/test_ppotrainer_gym.py`

### 3.1.1. Experiment Result
![PPO_CartPole](/docs/experiment_results/ppo/PPO_CartPole-v1.png "PPO_CartPole")

## 3.2. Gym Atari BeamRiderNoFrameskip-v4
[BeamRider Reference](https://github.com/openai/atari-py)

### 3.2.1. Reproduce script
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

### 3.2.2. Experiment Result
![PPO_BreamRider](/docs/experiment_results/ppo/PPO_BeamRider.png "PPO_BreamRider")

Important reference indicators
1. Episode Reward: According to your reward engine, the bigger the better
2. explained_variance: It reflects the difference between your critic VF_OUT and VALUE_TARGET; In general, the closer to 1, the better

## 3.3. Google Football
TODO


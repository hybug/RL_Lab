<!--
 * @Author: hanyu
 * @Date: 2021-06-15 10:34:08
 * @LastEditTime: 2023-03-10 17:08:46
 * @LastEditors: hanyu
 * @Description: work docs
 * @FilePath: /RL_Lab/docs/work_docs.md
-->
### 2021-06-16
#### 1. Done
1. Improve the display function in HG.
#### 2. Todo
1. Realize the reward function: currently 201, 301...
2. Test the HG env reset-step-reset process.
3. History segments implement.

### 2021-06-17
#### 1. Done
1. Apply the reward function: currently +1(eat food), -0.1(nothing), +1(terminal)
2. History segments implement.
3. Pause existing development, imple Multi-agent Rllib environment.
#### 2. Todo
1. Legal action mask
2. Distributed Multi-Env
3. Rllib training environment.
4. Test the HG env reset-step-reset process.

### 2021-06-22
#### 1. Done
1. Imple Multi-agent RLllib environments.
#### 2. Todo
1. PPO algorithm based on Ray distribution cluster.
2. Hungery-Geese Environment using ray-ppo.
3. policy_model.py warp_model()

### 2021-06-28
#### 1. Done
1. Start training rllib-model, command: `python examples/PPO_hungry_geese/rllib_ppo/train.py --exper_name rllib-hungrygeese-agent_4-gamma_0.99-reuse_3 --ray_workers 15 --rollout_fragment_length 2000 --gamma 0.99 --num_gpus 1 --train_batch_size 30000 --checkpoint_freq 200 --num_sgd_iter 3 --num_policies 4`
2. Decorator for getting&setting Tensorflow models' weights
3. Two utils functions: `get_tensor_shape`, `draw_categorical_samples`
#### 2. Todo
1. PPO algorithm based on Ray distribution cluster.
2. Hungery-Geese Environment using ray-ppo.
3. Finish Actor-Net and Critic-Net
4. policy_model.py warp_model()

### 2021-07-08
#### 1. Done
1. Submit the submission.py to kaggle and fix a series problem, scores: 600
2. Continue training rllib mutil-agent hungry-geese, `PYTHONPATH=./ python examples/PPO_hungry_geese/rllib_ppo/train.py --exper_name rllib-hungrygeese-agent_4-gamma_0.99-reuse_1 --ray_workers 15 --rollout_fragment_length 2000 --gamma 0.99 --num_gpus 1 --train_batch_size 30000 --checkpoint_freq 200 --num_sgd_iter 1 --num_policies 4 --model_restore /home/jj/workspace/hanyu/RL_Lab/result/hungry_geese/rllib-hungrygeese-agent_4-gamma_0.99-reuse_3/PPO_HungryGeeseEnv_1c317_00000_0_2021-06-30_07-09-28/checkpoint_002000/checkpoint-2000`

#### 2. Todo
1. PPO algorithm based on Ray distribution cluster.
2. Hungery-Geese Environment using ray-ppo.
3. Finish Actor-Net and Critic-Net
4. policy_model.py warp_model()

### 2021-07-09
#### 1. Done
1. PPO alogrithm picture
2. actor net & critic net

#### 2. Todo
1. PPO algorithm based on Ray distribution cluster.
2. Hungery-Geese Environment using ray-ppo.
3. Finish Actor-Net and Critic-Net
4. policy_model.py warp_model()


### 2021-07-12
#### 1. Done
1. Reward Function V2: increase death penalty. `PYTHONPATH=./ python examples/PPO_hungry_geese/rllib_ppo/train.py --exper_name rllib-hungrygeese-agent_4-gamma_0.99-reuse_1-reward_v2 --ray_workers 15 --rollout_fragment_length 2000 --gamma 0.99 --num_gpus 1 --train_batch_size 30000 --checkpoint_freq 200 --num_sgd_iter 1 --num_policies 4 --model_restore /home/jj/workspace/hanyu/RL_Lab/result/hungry_geese/rllib-hungrygeese-agent_4-gamma_0.99-reuse_3/PPO_HungryGeeseEnv_1c317_00000_0_2021-06-30_07-09-28/checkpoint_001400/checkpoint-1400`
2. Adjust gamma: `PYTHONPATH=./ python examples/PPO_hungry_geese/rllib_ppo/train.py --exper_name rllib-hungrygeese-agent_4-gamma_0.8-reuse_1-reward_v2 --ray_workers 15 --rollout_fragment_length 2000 --gamma 0.8 --num_gpus 1 --train_batch_size 30000 --checkpoint_freq 200 --num_sgd_iter 1 --num_policies 4 --model_restore /home/jj/workspace/hanyu/RL_Lab/result/hungry_geese/rllib-hungrygeese-agent_4-gamma_0.99-reuse_3/PPO_HungryGeeseEnv_1c317_00000_0_2021-06-30_07-09-28/checkpoint_001400/checkpoint-1400`

#### 2. Todo
1. PPO algorithm based on Ray distribution cluster.
2. Hungery-Geese Environment using ray-ppo.
3. Finish Actor-Net and Critic-Net
4. policy_model.py warp_model()

### 2021-07-13
#### 1. Done
1. Training from zero: `PYTHONPATH=./ python examples/PPO_hungry_geese/rllib_ppo/train.py --exper_name rllib-hungrygeese-agent_4-gamma_0.8-reuse_1-reward_v2 --ray_workers 15 --rollout_fragment_length 2000 --gamma 0.8 --num_gpus 1 --train_batch_size 30000 --checkpoint_freq 200 --num_sgd_iter 1 --num_policies 4`
#### 2. Todo
1. PPO algorithm based on Ray distribution cluster.
2. Hungery-Geese Environment using ray-ppo.
3. Finish Actor-Net and Critic-Net
4. policy_model.py warp_model()

### 2021-09-10
#### 1. Done
1. Sorry for not updating the content for a long time. Because my work is relatively busy, and the content of the work (Mahjong AI) seems to have some progress. I hope I can share with us then.
2. Hungery-Geese game has finished. My above simple job ranks 666.
#### 2. Todo
1. New competition! https://www.kaggle.com/c/lux-ai-2021/overview
2. I try to start using reinforcement learning to solve LordAI’s problems. My working stuffs on another project, it will be synchronized later

### 2022-07-25
#### 1. Done
Copy https://github.com/jw1401/PPO-Tensorflow-2.0
#### 2. Todo
Reproducing https://towardsdatascience.com/reproducing-google-research-football-rl-results-ac75cf17190e

### 2022-08-04
#### 1. Done
- [x] Reproduced Gym-CartPole-v1
- [x] Fix the wrong gae calculation
#### 2. Todo
- [ ] Reproducing https://towardsdatascience.com/reproducing-google-research-football-rl-results-ac75cf17190e
- [ ] Restructure the RolloutWorker.rollout()[Using Rllib SampleBatch & SampleBatchBuilder for better visualization in debug]
- [ ] There seems to be a problem in explaining variance calculation. The convergence in gfootball is not as good as that in benchwork, ev calculated by prev_value&value_target or curr_value&value_target?
- [ ] Optimize the logger output representation
- [ ] Check the efficiency of samplebatch

### 2022-08-05
#### 1. Done
- [x] Restructure the RolloutWorker.rollout()[Using Rllib SampleBatch & SampleBatchBuilder for better visualization in debug]
- [x] Optimize the logger output representation
- [x] Check the efficiency of samplebatch: Fixed, the GPU is not available
#### 2. Todo
- [ ] Finish Gfootbal's academy scenarios experiments
- [ ] Reproducing https://towardsdatascience.com/reproducing-google-research-football-rl-results-ac75cf17190e
- [ ] Check the ExplianceVariance calculation

### 2022-08-19
#### 1. Done
- [x] Reproduced the CartPole-v1 with SampleBatch-Version
#### 2. Todo
- [ ] Reproducing reinforcemenr learning benchmarks: [BeamRider, Breakout, Qbert, SpaceInvaders]
- [ ] Add the timestep variable into Tensorflow scale & logging info
- [ ] Fix the zero represatation of Episode Reward in logging info 
- [ ] Finish Gfootbal's academy scenarios experiments
- [ ] Reproducing https://towardsdatascience.com/reproducing-google-research-football-rl-results-ac75cf17190e
- [ ] Check the ExplianceVariance calculation

### 2022-08-26
#### 1. Done
- [x] Reproducing reinforcemenr learning benchmarks: [BeamRider], tricks&fixed bugs:
  - DeepMind atari env, according atari_wrappers[refer to rllib source code]
  - clip the reward to [-1, 1], using np.sign
  - convert lr rate from 5e-4(cant converge) to 5e-5, epsilon from 1e-5 to 1e-7
  - fix the value_fn_out shape error
  - fix the explained variance error calculation
- [x] Add the timestep variable into Tensorflow scale & logging info
- [x] Fix the zero represatation of Episode Reward in logging info 
- [x] Check the ExplianceVariance calculation
#### 2. Todo
- [ ] Clean up code and merge into main branch
- [ ] Write the README.md's atari part
- [ ] Add mahjong game env
- [ ] Add ray distributed training mode
- [ ] Finish Gfootbal's academy scenarios experiments
- [ ] Reproducing https://towardsdatascience.com/reproducing-google-research-football-rl-results-ac75cf17190e


### 2023-03-10
After half a year wasted on a fail project, I return back to finish this repository. 
Begin with finishing ppo's benchworks, and find out the reason for not convergence in football.
<!--
 * @Author: hanyu
 * @Date: 2021-06-15 10:34:08
 * @LastEditTime: 2021-07-09 10:29:55
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
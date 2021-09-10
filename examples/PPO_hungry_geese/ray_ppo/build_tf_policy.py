'''
Author: hanyu
Date: 2021-07-13 10:24:46
LastEditTime: 2021-07-14 03:56:54
LastEditors: hanyu
Description: dynamic tf policy
FilePath: /RL_Lab/examples/PPO_hungry_geese/ray_ppo/build_tf_policy.py
'''
import gym
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils.annotations import override
from tensorflow.python.ops.gen_array_ops import guarantee_const_eager_fallback
from config import logger
from ray.rllib.policy.dynamic_tf_policy import DynamicTFPolicy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.framework import try_import_tf
from typing import Any, Dict, List, Tuple, Union


tf1, tf, tfv = try_import_tf()


def build_tf_policy(
        name: str,
        loss_fn,
        config,
        postprocess_fn=None,
        stats_fn=None,
        gradients_fn=None,
        extra_action_fetches_fn=None,
        before_init=None,
        before_loss_init=None):
    """Helper function for creating a dynamic tf policy at runtime.

    Functions will be run in this order to initialize the policy:
        1. Placeholder setup: postprocess_fn
        2. Loss init: loss_fn, stats_fn
        3. Optimizer init: optimizer_fn, gradients_fn, apply_gradients_fn,
                           grad_stats_fn

    This means that you can e.g., depend on any policy attributes created in
    the running of `loss_fn` in later functions such as `stats_fn`.

    In eager mode, the following functions will be run repeatedly on each
    eager execution: loss_fn, stats_fn, gradients_fn, apply_gradients_fn,
    and grad_stats_fn.

    This means that these functions should not define any variables internally,
    otherwise they will fail in eager mode execution. Variable should only
    be created in make_model (if defined).

    Args:
        name (str): Name of the policy (e.g., "PPOTFPolicy").
        loss_fn (Callable[[
            Policy, ModelV2, Type[TFActionDistribution], SampleBatch],
            Union[TensorType, List[TensorType]]]): Callable for calculating a
            loss tensor.
        get_default_config (Optional[Callable[[None], TrainerConfigDict]]):
            Optional callable that returns the default config to merge with any
            overrides. If None, uses only(!) the user-provided
            PartialTrainerConfigDict as dict for this Policy.
        postprocess_fn (Optional[Callable[[Policy, SampleBatch,
            Optional[Dict[AgentID, SampleBatch]], MultiAgentEpisode], None]]):
            Optional callable for post-processing experience batches (called
            after the parent class' `postprocess_trajectory` method).
        stats_fn (Optional[Callable[[Policy, SampleBatch],
            Dict[str, TensorType]]]): Optional callable that returns a dict of
            TF tensors to fetch given the policy and batch input tensors. If
            None, will not compute any stats.
        optimizer_fn (Optional[Callable[[Policy, TrainerConfigDict],
            "tf.keras.optimizers.Optimizer"]]): Optional callable that returns
            a tf.Optimizer given the policy and config. If None, will call
            the base class' `optimizer()` method instead (which returns a
            tf1.train.AdamOptimizer).
        gradients_fn (Optional[Callable[[Policy,
            "tf.keras.optimizers.Optimizer", TensorType], ModelGradients]]):
            Optional callable that returns a list of gradients. If None,
            this defaults to optimizer.compute_gradients([loss]).
        apply_gradients_fn (Optional[Callable[[Policy,
            "tf.keras.optimizers.Optimizer", ModelGradients],
            "tf.Operation"]]): Optional callable that returns an apply
            gradients op given policy, tf-optimizer, and grads_and_vars. If
            None, will call the base class' `build_apply_op()` method instead.
        grad_stats_fn (Optional[Callable[[Policy, SampleBatch, ModelGradients],
            Dict[str, TensorType]]]): Optional callable that returns a dict of
            TF fetches given the policy, batch input, and gradient tensors. If
            None, will not collect any gradient stats.
        extra_action_out_fn (Optional[Callable[[Policy],
            Dict[str, TensorType]]]): Optional callable that returns
            a dict of TF fetches given the policy object. If None, will not
            perform any extra fetches.
        extra_learn_fetches_fn (Optional[Callable[[Policy],
            Dict[str, TensorType]]]): Optional callable that returns a dict of
            extra values to fetch and return when learning on a batch. If None,
            will call the base class' `extra_compute_grad_fetches()` method
            instead.
        validate_spaces (Optional[Callable[[Policy, gym.Space, gym.Space,
            TrainerConfigDict], None]]): Optional callable that takes the
            Policy, observation_space, action_space, and config to check
            the spaces for correctness. If None, no spaces checking will be
            done.
        before_init (Optional[Callable[[Policy, gym.Space, gym.Space,
            TrainerConfigDict], None]]): Optional callable to run at the
            beginning of policy init that takes the same arguments as the
            policy constructor. If None, this step will be skipped.
        before_loss_init (Optional[Callable[[Policy, gym.spaces.Space,
            gym.spaces.Space, TrainerConfigDict], None]]): Optional callable to
            run prior to loss init. If None, this step will be skipped.
        after_init (Optional[Callable[[Policy, gym.Space, gym.Space,
            TrainerConfigDict], None]]): Optional callable to run at the end of
            policy init. If None, this step will be skipped.
        make_model (Optional[Callable[[Policy, gym.spaces.Space,
            gym.spaces.Space, TrainerConfigDict], ModelV2]]): Optional callable
            that returns a ModelV2 object.
            All policy variables should be created in this function. If None,
            a default ModelV2 object will be created.
        action_sampler_fn (Optional[Callable[[TensorType, List[TensorType]],
            Tuple[TensorType, TensorType]]]): A callable returning a sampled
            action and its log-likelihood given observation and state inputs.
            If None, will either use `action_distribution_fn` or
            compute actions by calling self.model, then sampling from the
            so parameterized action distribution.
        action_distribution_fn (Optional[Callable[[Policy, ModelV2, TensorType,
            TensorType, TensorType],
            Tuple[TensorType, type, List[TensorType]]]]): Optional callable
            returning distribution inputs (parameters), a dist-class to
            generate an action distribution object from, and internal-state
            outputs (or an empty list if not applicable). If None, will either
            use `action_sampler_fn` or compute actions by calling self.model,
            then sampling from the so parameterized action distribution.
        mixins (Optional[List[type]]): Optional list of any class mixins for
            the returned policy class. These mixins will be applied in order
            and will have higher precedence than the DynamicTFPolicy class.
        get_batch_divisibility_req (Optional[Callable[[Policy], int]]):
            Optional callable that returns the divisibility requirement for
            sample batches. If None, will assume a value of 1.

    Returns:
        Type[DynamicTFPolicy]: A child class of DynamicTFPolicy based on the
            specified args.
    """

    if extra_action_fetches_fn is not None:
        extra_action_out_fn = extra_action_fetches_fn

    class policy_cls(DynamicTFPolicy):

        def __init__(self,
                     obs_space: gym.spaces.Space,
                     action_space: gym.space.Space,
                     config: Dict,
                     existing_model: ModelV2 = None,
                     existing_inputs: Dict[str, 'tf1.placeholder'] = None):

            def before_loss_init_wrapper(policy):
                if extra_action_out_fn is None:
                    policy._extra_action_fetches = {}
                else:
                    policy._extra_action_fetches = extra_action_out_fn(policy)
                    # policy._extra_action_fetches = extra_action_out_fn(policy)

            DynamicTFPolicy.__init__(self,
                                     obs_space=obs_space,
                                     action_space=action_space,
                                     config=config,
                                     loss_fn=loss_fn,
                                     stats_fn=stats_fn,
                                     grad_stats_fn=None,
                                     before_loss_init=before_loss_init_wrapper,
                                     make_model=None,
                                     action_sampler_fn=None,
                                     action_distribuution_fn=None,
                                     existing_inputs=existing_inputs,
                                     existing_model=existing_model,
                                     get_batch_divisibility_req=None)

            self.global_timestep = 0

        @override(Policy)
        def postprocess_trajectory(self,
                                   sample_batch,
                                   other_agent_batches=None,
                                   episode=None):
            # Call super's postprocess_trajectory first.
            sample_batch = Policy.postprocess_trajectory(self, sample_batch)
            if postprocess_fn:
                # Custom postprocess sample_batch
                # In ppo, that is compute_gae_for_sample_batch
                return postprocess_fn(self, sample_batch, other_agent_batches, episode)
            return sample_batch

        @override(TFPolicy)
        def optimizer(self):
            # TODO
            pass

        @override(TFPolicy)
        def gradients(self, optimizer, loss):
            # TODO
            pass

        @override(TFPolicy)
        def build_apply_op(self, optimizer, grads_and_vars):
            # TODO
            pass

        @override(TFPolicy)
        def extra_compute_action_fetches(self):
            # TODO
            pass

        @override(TFPolicy)
        def extra_compute_grad_fetches(self):
            # TODO
            pass

    def with_updates(**overrides):
        """Allows creating a TFPolicy cls based on settings of another one.

        Keyword Args:
            **overrides: The settings (passed into `build_tf_policy`) that
                should be different from the class that this method is called
                on.

        Returns:
            type: A new TFPolicy sub-class.

        Examples:
        >> MySpecialDQNPolicyClass = DQNTFPolicy.with_updates(
        ..    name="MySpecialDQNPolicyClass",
        ..    loss_function=[some_new_loss_function],
        .. )
        """
        # TODO
        pass

    def as_eager():
        return

    return policy_cls

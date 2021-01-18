'''
Author: hanyu
Date: 2021-01-11 13:14:01
LastEditTime: 2021-01-18 13:13:56
LastEditors: hanyu
Description: important sampling
FilePath: /test_ppo/module/ImportantSampling.py
'''
import tensorflow as tf


def important_sampling(log_p, log_old_p=None, clip_1=None, clip_2=None):
    '''
    description: get the important sampling ratio from log_p*log_old_p
    param {*}
    return {*}
    '''
    if log_old_p is None:
        log_old_p = log_p

    log_ratio = log_p - \
        tf.stop_gradient(tf.maximum(log_old_p, tf.math.log(1e-8)))
    ratio = tf.exp(log_ratio)
    with tf.name_scope('IS'):
        with tf.name_scope('log_p'):
            tf.summary.scalar('mean', tf.reduce_mean(log_p))
            tf.summary.scalar('l1_norm', tf.reduce_mean(tf.abs(log_p)))
            tf.summary.scalar('l2_norm', tf.sqrt(tf.reduce_mean(log_p ** 2)))
        with tf.name_scope('log_old_p'):
            tf.summary.scalar("mean", tf.reduce_mean(log_old_p))
            tf.summary.scalar("l1_norm", tf.reduce_mean(tf.abs(log_old_p)))
            tf.summary.scalar("l2_norm", tf.sqrt(
                tf.reduce_mean(log_old_p ** 2)))
        with tf.name_scope('ros'):
            tf.summary.scalar('mean', tf.reduce_mean(ratio))
            tf.summary.scalar("l1_norm", tf.reduce_mean(tf.abs(ratio)))
            tf.summary.scalar("l2_norm", tf.sqrt(tf.reduce_mean(ratio ** 2)))

    if clip_1 is not None:
        ratio = tf.maximum(ratio, clip_1)
    if clip_2 is not None:
        ratio = tf.minimum(ratio, clip_2)
    return ratio


def IS_from_logits(policy_logits, action, behavior_logits=None, clip_1=None, clip_2=None):
    log_p = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=action, logits=policy_logits
    )
    if behavior_logits is None:
        behavior_logits = policy_logits
    log_old_p = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=action, log_old_p=behavior_logits
    )
    ratio = important_sampling(
        log_p=log_p, log_old_p=log_old_p, clip_1=clip_1, clip_2=clip_2)
    return ratio

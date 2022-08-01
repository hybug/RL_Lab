'''
Author: hanyu
Date: 2022-07-19 11:35:25
LastEditTime: 2022-07-29 16:37:43
LastEditors: hanyu
Description: get logger
FilePath: /RL_Lab/get_logger.py
'''

import os
from dataclasses import dataclass
from datetime import datetime

import tensorflow as tf
from loguru import logger

BASEDIR = os.path.abspath(os.path.dirname(__file__))

logger.add(
    BASEDIR + '/logs/log_' + '_{time:YYYY-MM-DD}.log',
    level='INFO',
    format=
    '[{time:YYYY-MM-DDTHH:mm:ss}] [{module}:{line}] [{level}]: {message}',
    rotation='00:00')


@dataclass
class LoggerParams:

    experiment_name: str = ""
    log_dir: str = ""
    datetime_now: str = str(datetime.now())
    file_name: str = ""

    def __post_init__(self):
        c = ":-"
        for char in c:
            self.datetime_now = self.datetime_now.replace(char, "-")
        if self.experiment_name:
            self.file_name = self.experiment_name


class TFLogger():
    def __init__(self, experiment_name: str = "", log_dir: str = "") -> None:
        datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = experiment_name + "_" + datetime_str
        self.logger_params = LoggerParams(experiment_name=experiment_name,
                                          log_dir=log_dir)
        self.metrics = dict()
        self.summary_writer = tf.summary.create_file_writer(
            self.logger_params.log_dir +
            f'/logs/_TensorFlow_Training/{datetime_str}/{self.logger_params.file_name}'
        )

    def store(self, name=None, value=None):
        if value is not None:
            if name not in self.metrics.keys():
                self.metrics[name] = tf.keras.metrics.Mean(name=name)
            self.metrics[name].update_state(value)

    def log_metrics(self, epoch: int):
        logger.info('MEAN METRICS START')
        logger.info(f'Epoch: {epoch}')

        flag = False
        if not self.metrics:
            logger.info("No metrics")
        else:
            if "Episode Reward" in self.metrics.keys(
            ) and self.metrics["Episode Reward"] != 0:
                flag = True
            for key, metric in self.metrics.items():
                value = metric.result()
                if flag:
                    logger.info(f'{key}: {value}')

                if key not in ["Episode Reward", "Episode Length"
                               ] or value != 0:
                    with self.summary_writer.as_default():
                        tf.summary.scalar(key, value, step=epoch)
                metric.reset_states()
        logger.info("MEAN METRICS END")

'''
Author: hanyu
Date: 2021-07-13 10:08:40
LastEditTime: 2021-07-13 10:20:38
LastEditors: hanyu
Description: 
FilePath: /RL_Lab/config.py
'''
import os
import logging

BASEDIR = os.path.abspath(os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(process)d] [%(filename)s:%(lineno)d] [%(levelname)s]: %(message)s",
                    datefmt="[%Y-%m-%dT%H:%M:%S]")
logger = logging.getLogger(__name__)
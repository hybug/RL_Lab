'''
Author: hanyu
Date: 2021-03-30 12:22:56
LastEditTime: 2021-03-30 12:24:50
LastEditors: hanyu
Description: config file
FilePath: /MahjongWuHan-SL/config.py
'''
import os
import logging

BASEDIR = os.path.abspath(os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(process)d] [%(filename)s:%(lineno)d] [%(levelname)s]: %(message)s",
                    datefmt="[%Y-%m-%dT%H:%M:%S]",
                    # filename=BASEDIR + '/logs/log.txt',
                    filemode='a'
                    )
logger = logging.getLogger(__name__)
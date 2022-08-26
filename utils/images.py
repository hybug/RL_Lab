'''
Author: hanyu
Date: 2022-08-23 14:37:58
LastEditTime: 2022-08-23 14:39:02
LastEditors: hanyu
Description: images utils
FilePath: /RL_Lab/utils/images.py
'''
import numpy as np
import logging

logger = logging.getLogger(__name__)
try:
    import cv2

    cv2.ocl.setUseOpenCL(False)

    logger.debug("CV2 found for image processing.")
except ImportError:
    cv2 = None

if cv2 is None:
    try:
        from skimage import color, transform

        logger.debug("CV2 not found for image processing, using Skimage.")
    except ImportError:
        raise ModuleNotFoundError("Either scikit-image or opencv is required")


def resize(img: np.ndarray, height: int, width: int) -> np.ndarray:
    if cv2:
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return transform.resize(img, (height, width))


def rgb2gray(img: np.ndarray) -> np.ndarray:
    if cv2:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return color.rgb2gray(img)

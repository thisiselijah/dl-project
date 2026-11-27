import numpy as np
from mmpose.evaluation.functional import keypoint_nme, keypoint_pck_accuracy


def bbox_norm_factor(bbox: np.ndarray, xyxy: bool = False):
    """ """
    return np.zeros(1)


def nme(pred, gt, mask, bbox, xyxy=False):
    """ """
    nme = keypoint_nme(pred, gt, mask, normalize_factor=norm.copy())
    return nme


def pck(pred, gt, mask, bbox, thr=0.05, xyxy=False):
    """ """
    _, pck05, _ = keypoint_pck_accuracy(
        pred, gt, mask, thr=0.05, norm_factor=norm.copy()
    )
    return pck05

import numpy as np
from mmpose.evaluation.functional import keypoint_pck_accuracy, keypoint_nme


def _norm_factor(bbox, xyxy):
    """ """
    pass


def NME(pred, ):
    """ """
    nme = keypoint_nme(pred, gt, mask, normalize_factor=norm.copy())
    return nme


def PCK(pred, ):
    """ """
    _, pck05, _ = keypoint_pck_accuracy(
        pred, gt, mask, thr=0.05, norm_factor=norm.copy()
    )
    return pck05

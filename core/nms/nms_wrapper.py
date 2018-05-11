# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from config import cfg
from .cpu_nms import cpu_nms

def nms(dets, thresh, force_cpu=False):
    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS and not force_cpu:
        raise NotImplementedError('gpu_nms is not implemented.')
    else: return cpu_nms(dets, thresh)

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Zhuang Liu
# --------------------------------------------------------

import numpy as np


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0]
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1]
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
    ex_p1_x = ex_rois[:, 4]
    ex_p1_y = ex_rois[:, 5]
    ex_p2_x = ex_rois[:, 6]
    ex_p2_y = ex_rois[:, 7]
    ex_p3_x = ex_rois[:, 8]
    ex_p3_y = ex_rois[:, 9]
    ex_p4_x = ex_rois[:, 10]
    ex_p4_y = ex_rois[:, 11]

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0]
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1]
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
    gt_p1_x = gt_rois[:, 4]
    gt_p1_y = gt_rois[:, 5]
    gt_p2_x = gt_rois[:, 6]
    gt_p2_y = gt_rois[:, 7]
    gt_p3_x = gt_rois[:, 8]
    gt_p3_y = gt_rois[:, 9]
    gt_p4_x = gt_rois[:, 10]
    gt_p4_y = gt_rois[:, 11]

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)
    targets_dp1x = (gt_p1_x - ex_p1_x) / ex_widths
    targets_dp1y = (gt_p1_y - ex_p1_y) / ex_heights
    targets_dp2x = (gt_p2_x - ex_p2_x) / ex_widths
    targets_dp2y = (gt_p2_y - ex_p2_y) / ex_heights
    targets_dp3x = (gt_p3_x - ex_p3_x) / ex_widths
    targets_dp3y = (gt_p3_y - ex_p3_y) / ex_heights
    targets_dp4x = (gt_p4_x - ex_p4_x) / ex_widths
    targets_dp4y = (gt_p4_y - ex_p4_y) / ex_heights

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh, targets_dp1x, targets_dp1y, targets_dp2x, targets_dp2y,
         targets_dp3x, targets_dp3y, targets_dp4x, targets_dp4y)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    p1_x = boxes[:, 4]
    p1_y = boxes[:, 5]
    p2_x = boxes[:, 6]
    p2_y = boxes[:, 7]
    p3_x = boxes[:, 8]
    p3_y = boxes[:, 9]
    p4_x = boxes[:, 10]
    p4_y = boxes[:, 11]

    dx = deltas[:, 0::12]
    dy = deltas[:, 1::12]
    dw = deltas[:, 2::12]
    dh = deltas[:, 3::12]
    dp1x = deltas[:, 4::12]
    dp1y = deltas[:, 5::12]
    dp2x = deltas[:, 6::12]
    dp2y = deltas[:, 7::12]
    dp3x = deltas[:, 8::12]
    dp3y = deltas[:, 9::12]
    dp4x = deltas[:, 10::12]
    dp4y = deltas[:, 11::12]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    pred_p1x = dp1x * widths[:, np.newaxis] + p1_x[:, np.newaxis]
    pred_p1y = dp1y * heights[:, np.newaxis] + p1_y[:, np.newaxis]
    pred_p2x = dp2x * widths[:, np.newaxis] + p2_x[:, np.newaxis]
    pred_p2y = dp2y * heights[:, np.newaxis] + p2_y[:, np.newaxis]
    pred_p3x = dp3x * widths[:, np.newaxis] + p3_x[:, np.newaxis]
    pred_p3y = dp3y * heights[:, np.newaxis] + p3_y[:, np.newaxis]
    pred_p4x = dp4x * widths[:, np.newaxis] + p4_x[:, np.newaxis]
    pred_p4y = dp4y * heights[:, np.newaxis] + p4_y[:, np.newaxis]


    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # xmin
    pred_boxes[:, 0::12] = pred_ctr_x - 0.5 * pred_w
    # ymin
    pred_boxes[:, 1::12] = pred_ctr_y - 0.5 * pred_h
    # xmax
    pred_boxes[:, 2::12] = pred_ctr_x + 0.5 * pred_w
    # ymax
    pred_boxes[:, 3::12] = pred_ctr_y + 0.5 * pred_h
    # x1
    pred_boxes[:, 4::12] = pred_p1x
    # y1
    pred_boxes[:, 5::12] = pred_p1y
    # x2
    pred_boxes[:, 6::12] = pred_p2x
    # y2
    pred_boxes[:, 7::12] = pred_p2y
    # x3
    pred_boxes[:, 8::12] = pred_p3x
    # y3
    pred_boxes[:, 9::12] = pred_p3y
    # x4
    pred_boxes[:, 10::12] = pred_p4x
    # y4
    pred_boxes[:, 11::12] = pred_p4y

    return pred_boxes


def clip_boxes(boxes, im_shape):
    # x1 >= 0, x2 < im_shape[1]
    boxes[:, 0:-1:2] = np.maximum(np.minimum(boxes[:, 0:-1:2], im_shape[1] - 1), 0)
    # y1 >= 0, y2 < im_shape[0]
    boxes[:, 1::2] = np.maximum(np.minimum(boxes[:, 1::2], im_shape[0] - 1), 0)

    return boxes

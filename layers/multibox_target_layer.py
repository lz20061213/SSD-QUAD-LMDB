# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# Modified by Zhuang Liu
# --------------------------------------------------------

import dragon.core.workspace as ws
import dragon.vm.caffe as caffe
import numpy as np
from six.moves import xrange

from config import cfg
from core.utils.bbox_transform import bbox_transform


class MultiBoxTargetLayer(caffe.Layer):
    def _fetch_gt_boxes(self, annotations):
        all_gt_boxes = {}
        for annotation in annotations:
            item_id = int(annotation[0])
            if not item_id in all_gt_boxes:
                all_gt_boxes[item_id] = []
            all_gt_boxes[item_id].append(annotation[1:])
        return all_gt_boxes

    def _compute_targets(self, ex_rois, gt_rois):
        targets = bbox_transform(ex_rois, gt_rois)
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
            # optionally normalize targets by dividing a precomputed stddev
            # which was used in RCNN to prevent the grads of bbox from vanishing early
            targets = targets / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        return np.hstack(
            (gt_rois[:, 12, np.newaxis], targets)).astype(np.float32, copy=False)  # by lz.

    def forward(self, bottom, top):
        # fetch matches between default boxes and gt boxes
        all_match_inds = ws.FetchTensor(bottom[0])

        # fetch the labels (after hard mining possibly)
        all_match_labels = ws.FetchTensor(bottom[1])

        # fetch the default boxes(anchors)
        prior_boxes = ws.FetchTensor(bottom[2])

        # fetch the annotations
        annotations = ws.FetchTensor(bottom[3])

        # decode gt boxes from annotations
        all_gt_boxes = self._fetch_gt_boxes(annotations)

        num_images = len(all_gt_boxes)
        num_priors = len(prior_boxes)

        all_bbox_targets = np.zeros((num_images, num_priors, 12), dtype=np.float32)
        all_bbox_inside_weights = np.zeros(all_bbox_targets.shape, dtype=np.float32)
        all_bbox_outside_weights = np.zeros(all_bbox_targets.shape, dtype=np.float32)

        # number of matched boxes(#positive)
        # we divide it by ``IMS_PER_BATCH`` as SmoothLLLoss will divide it also
        bbox_normalization = len(np.where(all_match_labels > 0)[0]) / cfg.TRAIN.IMS_PER_BATCH

        for im_idx in xrange(num_images):
            match_inds = all_match_inds[im_idx]
            match_labels = all_match_labels[im_idx]
            gt_boxes = np.array(all_gt_boxes[im_idx], dtype=np.float32)

            # sample fg-rois(default boxes) & gt-rois(gt boxes)
            ex_inds = np.where(match_labels > 0)[0]
            ex_rois = prior_boxes[ex_inds]
            gt_assignment = match_inds[ex_inds].astype(np.int32, copy=False)
            gt_rois = gt_boxes[gt_assignment]

            # compute fg targets
            targets = self._compute_targets(ex_rois, gt_rois)

            # assign targets & inside weights & outside weights
            all_bbox_targets[im_idx][ex_inds] = targets[:, 1:]
            all_bbox_inside_weights[im_idx, :] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
            all_bbox_outside_weights[im_idx][ex_inds] = 1.0 / bbox_normalization

        # feed bbox targets to compute bbox regression loss
        ws.FeedTensor(top[0], all_bbox_targets)

        # feed inside weights for SmoothL1Loss
        ws.FeedTensor(top[1], all_bbox_inside_weights)

        # feed outside weights for SmoothL1Loss
        ws.FeedTensor(top[2], all_bbox_outside_weights)


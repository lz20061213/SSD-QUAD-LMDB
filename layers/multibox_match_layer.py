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

from core.utils.cython_bbox import bbox_overlaps


class MultiBoxMatchLayer(caffe.Layer):
    def setup(self, bottom, top):
        layer_params = eval(self.param_str)
        self._num_classes = layer_params.get('num_classes', 0)
        # the lowest IoU for positive boxes
        self._threshold = layer_params.get('overlap_threshold', 0.5)
        # ignore the boundary boxes(boxes that are outside the image) if True
        self._ignore_cross_boundary_bbox = \
            layer_params.get('ignore_cross_boundary_bbox', False)
        self._match_type = layer_params.get('match_type', 'PER_PREDICTION')

    def _fetch_gt_boxes(self, annotations):
        all_gt_boxes = {}
        for annotation in annotations:
            item_id = int(annotation[0])
            if not all_gt_boxes.has_key(item_id):
                all_gt_boxes[item_id] = []
            all_gt_boxes[item_id].append(annotation[1:])
        return all_gt_boxes

    def forward(self, bottom, top):
        # fetch the default boxes(anchors)
        prior_boxes = ws.FetchTensor(bottom[0])

        # fetch the annotations
        annotations = ws.FetchTensor(bottom[1])

        # decode gt boxes from annotations
        all_gt_boxes = self._fetch_gt_boxes(annotations)
        num_images = len(all_gt_boxes)
        num_priors = len(prior_boxes)

        # do matching between prior boxes and gt boxes
        # we simply use ``0`` as background id
        all_match_inds = np.empty((num_images, num_priors), dtype=np.float32)
        all_match_inds.fill(-1)
        all_match_labels = np.zeros(all_match_inds.shape, dtype=np.float32)
        all_max_overlaps = np.zeros(all_match_inds.shape, dtype=np.float32)

        for im_idx in xrange(num_images):
            # gt_boxes shape: [num_objects, {norm_x1, norm_y1, norm_x2, norm_y2,
            # norm_px1, norm_py1, norm_px2, norm_py2, norm_px3, norm_py3, norm_px4, norm_py4, cls}]
            # we ensure that each image has at least one ground-truth box
            gt_boxes = all_gt_boxes.get(im_idx, None)
            assert gt_boxes is not None
            if isinstance(gt_boxes, list): gt_boxes = np.array(gt_boxes, dtype=np.float32)

            # compute the overlaps between prior boxes and gt boxes
            # by lz.
            overlaps = bbox_overlaps(
                np.ascontiguousarray(prior_boxes[:, :4], dtype=np.float),
                np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(num_priors), argmax_overlaps]
            all_max_overlaps[im_idx] = max_overlaps

            # 1. bipartite matching & assignments (introduced by MultiBox)
            bipartite_inds = overlaps.argmax(axis=0)
            class_assignment = gt_boxes[:, 12]  # by lz.
            all_match_inds[im_idx][bipartite_inds] = np.arange(gt_boxes.shape[0])
            all_match_labels[im_idx][bipartite_inds] = class_assignment

            # 2. per prediction matching & assignments (optional | default)
            # note that SSD(by W.Liu) match each prior box for only once
            # we simply implement it by clobbering the assignments matched in [1]
            if self._match_type == 'PER_PREDICTION':
                per_inds = np.where(max_overlaps >= self._threshold)[0]
                gt_assignment = argmax_overlaps[per_inds]
                class_assignment = gt_boxes[gt_assignment, 12]  # by lz.
                all_match_inds[im_idx][per_inds] = gt_assignment
                all_match_labels[im_idx][per_inds] = class_assignment

            # 3. clobber boundary_bbox as background if necessary
            if self._ignore_cross_boundary_bbox:
                outside_inds = np.where(
                    (prior_boxes[:, 0] < 0) |
                    (prior_boxes[:, 1] < 0) |
                    (prior_boxes[:, 2] > 1) |
                    (prior_boxes[:, 3] > 1))[0]
                all_match_inds[im_idx][outside_inds] = -1
                all_match_labels[im_idx][outside_inds] = -1

        # feed the indices to make bbox targets
        ws.FeedTensor(top[0], all_match_inds)

        # feed the labels to do hard negative mining
        ws.FeedTensor(top[1], all_match_labels)

        # feed the max overlaps do hard negative mining
        if len(top) > 2:
            ws.FeedTensor(top[2], all_max_overlaps)

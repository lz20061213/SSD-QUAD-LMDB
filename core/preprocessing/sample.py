# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
import numpy.random as npr
from six.moves import xrange

from core.utils.bbox_transform import clip_boxes


class Sampler(object):
    def __init__(self, samplers):
        if not isinstance(samplers, list): samplers = [samplers]
        self._samplers = []
        for sampler in samplers:
            sample_param = sampler.get('sampler', None)
            max_trials = sampler.get('max_trials', None)
            max_sample = sampler.get('max_sample', None)
            if sample_param is None or \
                max_trials is None or \
                max_sample is None: continue
            self._samplers.append(sampler)

    def _compute_overlap(self, rand_box, query_box):
        overlap = 0.0
        iw = min(rand_box[2], query_box[2]) \
             - max(rand_box[0], query_box[0])
        if iw > 0:
            ih = min(rand_box[3], query_box[3]) \
                 - max(rand_box[1], query_box[1])
            if ih > 0:
                ua = float((rand_box[2] - rand_box[0]) * (rand_box[3] - rand_box[1])
                           + (query_box[2] - query_box[0]) * (query_box[3] - query_box[1])
                           - iw * ih)
                overlap = iw * ih / ua
        return overlap

    def _compute_overlaps(self, rand_box, gt_boxes):
        K = len(gt_boxes)
        overlaps = np.zeros((K,), dtype=np.float32)
        for k in range(K):
            box_area = (
                (gt_boxes[k, 2] - gt_boxes[k, 0]) *
                (gt_boxes[k, 3] - gt_boxes[k, 1]))
            iw = min(rand_box[2], gt_boxes[k, 2]) - \
                    max(rand_box[0], gt_boxes[k, 0])
            if iw > 0:
                ih = min(rand_box[3], gt_boxes[k, 3]) - \
                        max(rand_box[1], gt_boxes[k, 1])
                if ih > 0:
                    ua = float(
                            (rand_box[2] - rand_box[0]) *
                            (rand_box[3] - rand_box[1]) +
                            box_area - iw * ih)
                    overlaps[k] = iw * ih / ua
        return overlaps

    def _generate_sample(self, sample_param):
        min_scale = sample_param.get('min_scale', 1.0)
        max_scale = sample_param.get('max_scale', 1.0)
        scale = npr.uniform(min_scale, max_scale)
        min_aspect_ratio = sample_param.get('min_aspect_ratio', 1.0)
        max_aspect_ratio = sample_param.get('max_aspect_ratio', 1.0)
        min_aspect_ratio = max(min_aspect_ratio, scale**2)
        max_aspect_ratio = min(max_aspect_ratio, 1.0 / (scale**2))
        aspect_ratio = npr.uniform(min_aspect_ratio, max_aspect_ratio)
        bbox_w = scale * (aspect_ratio ** 0.5)
        bbox_h = scale / (aspect_ratio ** 0.5)
        w_off = npr.uniform(0.0, float(1 - bbox_w))
        h_off = npr.uniform(0.0, float(1 - bbox_h))
        return np.array([w_off, h_off, w_off + bbox_w, h_off + bbox_h])

    def _check_satisfy(self, sample_box, gt_boxes, constraint):
        min_jaccard_overlap = constraint.get('min_jaccard_overlap', None)
        max_jaccard_overlap = constraint.get('max_jaccard_overlap', None)
        if min_jaccard_overlap == None and \
            max_jaccard_overlap == None:
            return True

        for gt_box in gt_boxes:
            overlap = self._compute_overlap(sample_box, gt_box)
            if min_jaccard_overlap is not None:
                if overlap < min_jaccard_overlap: continue
            if max_jaccard_overlap is not None:
                if overlap > max_jaccard_overlap: continue
            return True

        return False

    def _generate_batch_samples(self, gt_boxes):
        sample_boxes = []
        for sampler in self._samplers:
            found = 0
            for i in xrange(sampler['max_trials']):
                if found >= sampler['max_sample']: break
                sample_box = self._generate_sample(sampler['sampler'])
                if 'sample_constraint' in sampler:
                    ok = self._check_satisfy(sample_box, gt_boxes,
                                               sampler['sample_constraint'])
                    if not ok: continue
                found += 1
                sample_boxes.append(sample_box)
        return sample_boxes

    def _rand_crop(self, im, rand_box, gt_boxes=None):
        im_h = im.shape[0]
        im_w = im.shape[1]
        w_off = int(rand_box[0] * im_w)
        h_off = int(rand_box[1] * im_h)
        crop_w = int((rand_box[2] - rand_box[0]) * im_w)
        crop_h = int((rand_box[3] - rand_box[1]) * im_h)

        new_im = im[h_off: h_off + crop_h, w_off: w_off + crop_w, :]

        if gt_boxes is not None:
            #ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2.0
            #ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2.0
            xmin = gt_boxes[:, 0]
            ymin = gt_boxes[:, 1]
            xmax = gt_boxes[:, 2]
            ymax = gt_boxes[:, 3]
            # keep the ground-truth box whose center is in the sample box
            # implement ``EmitConstraint.CENTER`` in the original SSD
            # by lz.
            # keep the ground-truth box whose min max point is in the sample box and
            #keep_inds = np.where((ctr_x >= rand_box[0]) & (ctr_x <= rand_box[2])
                                 #& (ctr_y >= rand_box[1]) & (ctr_y <= rand_box[3]))[0]
            keep_inds = np.where((xmin >= rand_box[0]) & (xmax <= rand_box[2])
                                 & (ymin >= rand_box[1]) & (ymax <= rand_box[3]))[0]
            gt_boxes = gt_boxes[keep_inds]
            new_gt_boxes = gt_boxes.astype(gt_boxes.dtype, copy=True)
            new_gt_boxes[:, 0:-1:2] = (gt_boxes[:, 0:-1:2] * im_w - w_off)
            new_gt_boxes[:, 1::2] = (gt_boxes[:, 1::2] * im_h - h_off)
            new_gt_boxes = clip_boxes(new_gt_boxes, (crop_h, crop_w))
            new_gt_boxes[:, 0:-1:2] = new_gt_boxes[:, 0:-1:2] / crop_w
            new_gt_boxes[:, 1::2] = new_gt_boxes[:, 1::2] / crop_h

            return new_im, new_gt_boxes

        return new_im, gt_boxes

    def sample_image(self, im, gt_boxes):
        sample_boxes = self._generate_batch_samples(gt_boxes)
        if len(sample_boxes) > 0:
            # apply sampling if found at least one valid sample box
            # then randomly pick one
            sample_idx = npr.randint(0, len(sample_boxes))
            rand_box = sample_boxes[sample_idx]
            im, gt_boxes = self._rand_crop(im, rand_box, gt_boxes)
        return im, gt_boxes
# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
from six.moves import xrange

import dragon.vm.caffe as caffe
import dragon.core.workspace as ws


class HardMiningLayer(caffe.Layer):
    def setup(self, bottom, top):
        layer_params = eval(self.param_str)
        self._neg_pos_ratio = layer_params.get('neg_pos_ratio', 3)
        self._neg_overlap = layer_params.get('neg_overlap', 0.5)

    def forward(self, bottom, top):
        # fetch the labels from the primary matches.
        all_match_labels = ws.FetchTensor(bottom[0])

        # fetch the max overlaps between default boxes and gt boxes
        all_max_overlaps = ws.FetchTensor(bottom[1])

        # fetch the confidences computed by SoftmaxLayer
        all_conf_prob = ws.FetchTensor(bottom[2])


        # label ``-1`` will be ignored
        all_labels = np.empty(all_match_labels.shape, dtype=np.float32)
        all_labels.fill(-1)

        for im_idx in xrange(all_match_labels.shape[0]):
            matche_labels = all_match_labels[im_idx]
            max_overlaps = all_max_overlaps[im_idx]

            # compute conf loss
            conf_prob = all_conf_prob[im_idx]
            conf_loss = np.zeros(matche_labels.shape, dtype=np.float32)
            inds = np.where(matche_labels >= 0)[0]
            flt_min = np.finfo(float).eps
            conf_loss[inds] = -1.0 * np.log(np.maximum(
                    conf_prob[inds, matche_labels[inds].astype(np.int32)], flt_min))

            # filter negatives
            fg_inds = np.where(matche_labels > 0)[0]
            neg_inds = np.where(matche_labels == 0)[0]
            neg_overlaps = max_overlaps[neg_inds]
            eligible_neg_inds = np.where(neg_overlaps < self._neg_overlap)[0]
            sel_inds = neg_inds[eligible_neg_inds]

            # do mining
            sel_loss = conf_loss[sel_inds]
            num_pos = len(fg_inds)
            num_sel = min(int(num_pos * self._neg_pos_ratio), len(sel_inds))
            sorted_sel_inds = sel_inds[np.argsort(-sel_loss)]
            bg_inds = sorted_sel_inds[:num_sel]
            all_labels[im_idx][fg_inds] = matche_labels[fg_inds] # keep fg indices
            all_labels[im_idx][bg_inds] = 0 # use hard negatives as bg indices

        # feed labels to compute cls loss
        ws.FeedTensor(top[0], all_labels)
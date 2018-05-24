# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# Modified by Zhuang Liu
# --------------------------------------------------------

import numpy as np

import dragon.vm.caffe as caffe
import dragon.core.workspace as ws

from .generate_anchors import generate_anchors


class PriorBoxLayer(caffe.Layer):
    """Generate default boxes(anchors).

    This layer may be redundant during inference time.

    """
    def setup(self, bottom, top):
        layer_params = eval(self.param_str)
        self._step = layer_params.get('step', None)
        self._offset = layer_params.get('offset', 0.5)
        self._clip = layer_params.get('clip', False)
        min_size = layer_params.get('min_size', [])
        max_size = layer_params.get('max_size', [])
        if not isinstance(min_size, list): min_size = [min_size]
        if not isinstance(max_size, list): max_size = [max_size]
        if len(max_size) > 0: assert len(min_size) == len(max_size)
        ratios = layer_params.get('aspect_ratio', [])
        aspect_ratios = [1.0]
        flip = layer_params.get('flip', True)
        for aspect_ratio in ratios:
            if not aspect_ratio in aspect_ratios:
                aspect_ratios.append(aspect_ratio)
                if flip: aspect_ratios.append(1.0 / aspect_ratio)
        self._num_anchors = len(min_size) * len(aspect_ratios) + len(max_size)
        self._anchors = generate_anchors(min_size, max_size, aspect_ratios)

    def forward(self, bottom, top):
        feature_maps = ws.FetchTensor(bottom[0])
        im_info = ws.FetchTensor(bottom[1])
        map_height, map_width = feature_maps.shape[2:]

        # 1. generate base grids
        shift_x = (np.arange(0, map_width) + self._offset) * self._step
        shift_y = (np.arange(0, map_height) + self._offset) * self._step
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        # by lz.
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # 2. apply anchors on base grids
        # add A anchors (1, A, 12) to
        # cell K shifts (K, 1, 12) to get
        # shift anchors (K, A, 12)
        # reshape to (K * A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]  # K = map_h * map_w
        # by lz.
        all_anchors = (self._anchors.reshape((1, A, 12)) +
                       shifts.reshape((1, K, 12)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 12)).astype(np.float32)

        all_anchors[:, 0::2] /= im_info[1] # normalize by width
        all_anchors[:, 1::2] /= im_info[0] # normalize by height

        # 3. clip if necessary (default is False)
        if self._clip:
            all_anchors = np.minimum(np.maximum(all_anchors, 0.0), 1.0)

        # feed the default boxes(anchors)
        ws.FeedTensor(top[0], all_anchors)
# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# Modified by Zhuang Liu
# --------------------------------------------------------

import numpy as np
from six.moves import xrange

def generate_anchors(min_sizes, max_sizes, ratios):
    """
    Generate anchor (reference) windows by enumerating
    aspect ratios, min_sizes, max_sizes wrt a reference ctr (x, y, w, h)
    Modified by lz. 2018/05/03
    (xmin, ymin, xmax, ymax, px1, py1, px2, py2, px3, py3, px4, py4)
    where px1 = xmin, py1 = (ymin+ymax) / 2, ...
    """


    total_anchors = []

    for idx, min_size in enumerate(min_sizes):
        # note that SSD assume it is a ctr-anchor
        base_anchor = np.array([0, 0, min_size, min_size])
        anchors = _ratio_enum(base_anchor, ratios)
        if len(max_sizes) > 0:
            max_size = max_sizes[idx]
            _anchors = anchors[0].reshape((1, 12))
            _anchors = np.vstack([_anchors, _max_size_enum(
                base_anchor, min_size, max_size)])
            anchors = np.vstack([_anchors, anchors[1:]])

        total_anchors.append(anchors)

    return np.vstack([total_anchors[i] for i in xrange(len(total_anchors))])


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    Note that it is a little different from Faster-RCNN
    """

    w = anchor[2]; h = anchor[3]
    x_ctr = anchor[0]; y_ctr = anchor[1]
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    # by lz.
    ps = np.ones(shape=ws.shape, dtype=np.float64)
    anchors = np.hstack((x_ctr - 0.5 * (ws),    # xmin
                         y_ctr - 0.5 * (hs),    # ymin
                         x_ctr + 0.5 * (ws),    # xmax
                         y_ctr + 0.5 * (hs),    # ymax
                         x_ctr - 0.5 * (ws),    # px1
                         y_ctr * (ps),          # py1
                         x_ctr * (ps),          # px2
                         y_ctr - 0.5 * (hs),    # py2
                         x_ctr + 0.5 * (ws),    # px3
                         y_ctr * (ps),          # py3
                         x_ctr * (ps),          # py4
                         y_ctr + 0.5 * (hs)))   # px4
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    hs = np.round(np.sqrt(size_ratios))
    ws = np.round(hs * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _max_size_enum(base_anchor, min_size, max_size):
    """
    Enumerate a anchor for max_size wrt base_anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(base_anchor)
    ws = hs = np.sqrt([min_size * max_size])
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':

    print(generate_anchors(min_sizes=[30], max_sizes=[60], ratios=[1, 0.5, 2]))

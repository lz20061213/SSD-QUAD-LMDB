# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# Modified by Zhuang Liu
# --------------------------------------------------------

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
# from config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 16

# Use shuffle after each epoch
__C.TRAIN.USE_SHUFFLE = True

# Use the difficult(under occlusion) objects
__C.TRAIN.USE_DIFF = True

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Normalize the targets (divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True

# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # by lz.

# Normalize the targets using "precomputed" (or made up) stddev
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)  # by lz.


__C.TRAIN.DISTORT_PARAM = {'brightness_prob': 0.5,
                           'contrast_prob': 0.5,
                           'saturation_prob': 0.5}

__C.TRAIN.EXPAND_PARAM = {'prob': 0.5,
                          'max_expand_ratio': 1.5}

__C.TRAIN.RESIZE_PARAM = {'height': 300,
                          'width': 300,
                          'interp_mode': ['LINEAR', 'AREA', 'NEAREST', 'CUBIC', 'LANCZOS4']}

__C.TRAIN.SAMPLERS = [{'sampler': {},
                       'max_trials': 1,
                       'max_sample': 1},

                      {'sampler': {'min_scale': 0.3,
                                   'max_scale': 1.0,
                                   'min_aspect_ratio': 0.5,
                                   'max_aspect_ratio': 2.0},
                       'sample_constraint': {'min_jaccard_overlap': 0.1},
                       'max_trials': 50,
                       'max_sample': 1},

                      {'sampler': {'min_scale': 0.3,
                                   'max_scale': 1.0,
                                   'min_aspect_ratio': 0.5,
                                   'max_aspect_ratio': 2.0},
                       'sample_constraint': {'min_jaccard_overlap': 0.3},
                       'max_trials': 50,
                       'max_sample': 1},

                      {'sampler': {'min_scale': 0.3,
                                   'max_scale': 1.0,
                                   'min_aspect_ratio': 0.5,
                                   'max_aspect_ratio': 2.0},
                       'sample_constraint': {'min_jaccard_overlap': 0.5},
                       'max_trials': 50,
                       'max_sample': 1},

                      {'sampler': {'min_scale': 0.3,
                                   'max_scale': 1.0,
                                   'min_aspect_ratio': 0.5,
                                   'max_aspect_ratio': 2.0},
                       'sample_constraint': {'min_jaccard_overlap': 0.7},
                       'max_trials': 50,
                       'max_sample': 1},

                      {'sampler': {'min_scale': 0.3,
                                   'max_scale': 1.0,
                                   'min_aspect_ratio': 0.5,
                                   'max_aspect_ratio': 2.0},
                       'sample_constraint': {'min_jaccard_overlap': 0.9},
                       'max_trials': 50,
                       'max_sample': 1},

                      {'sampler': {'min_scale': 0.3,
                                   'max_scale': 1.0,
                                   'min_aspect_ratio': 0.5,
                                   'max_aspect_ratio': 2.0},
                       'sample_constraint': {'max_jaccard_overlap': 1.0},
                       'max_trials': 50,
                       'max_sample': 1}
                      ]

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

__C.TRAIN.SNAPSHOT_FOLDER = ''

#
# Testing options
#

__C.TEST = edict()

__C.TEST.BATCH_SIZE = 1

__C.TEST.THRESH = 0.5

__C.TEST.NMS = 0.45

__C.TEST.NMS_TOP_K = 400

__C.TEST.MAX_PER_IMAGE = 200

#
# MISC
#

__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__)))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Name the active image database
__C.DATABASE = ''

# Model directory
__C.MODELS_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'models', 'pascal_voc'))

__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0


def get_output_dir(imdb, net=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', imdb.name))
    if net is not None:
        outdir = osp.join(outdir, net.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# Modified by Zhuang Liu
# --------------------------------------------------------

import os.path as osp
import numpy as np

import dragon.vm.caffe as caffe

from config import cfg, cfg_from_file
from core.train import train_net
from datasets.factory import get_imdb

from layers import *

imdb_name = 'plane_2018_trainval'
gpu_id = 1

solver_txt = 'models/quad/VGG16/300x300/solver.prototxt'
pretrained_model = 'data/imagenet_models/VGG16.reduce.caffemodel'
#snapshot_model = 'output/plane_2018_trainval/QUAD_VGG16_iter_70000.caffemodel'
snapshot_model = None
#start_iter = 70000
start_iter = 1
max_iters = 150000
warm_up = 0
import dragon.config
dragon.config.LogOptimizedGraph()
import dragon.memonger as opt
opt.ShareGrads()

cfg.DATA_DIR = 'data/quad'


if __name__ == '__main__':

    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)

    # setup caffe
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    # setup database
    cfg.DATABASE = imdb_name
    imdb = get_imdb(imdb_name)
    print 'Database({}): {} images will be used to train.'.format(cfg.DATABASE, imdb.db_size)
    output_dir = osp.abspath(osp.join(cfg.ROOT_DIR, 'output', imdb_name))
    print 'Output will be saved to `{:s}`'.format(output_dir)

    # train net
    train_net(solver_txt, output_dir,
              pretrained_model=pretrained_model,
              snapshot_model=snapshot_model,
              start_iter=start_iter,
              max_iters=max_iters,
              warm_up=warm_up)
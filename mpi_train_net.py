# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import os.path as osp
import numpy as np

import dragon.vm.caffe as caffe
import dragon.core.mpi as mpi
import dragon.memonger as opt

from config import cfg
from core.train import train_net
from datasets.factory import get_imdb

from layers import *

imdb_name = 'voc_0712_trainval'
gpus = [0, 1, 2, 3]

solver_txt = 'models/pascal_voc/VGG16/solver.prototxt'
pretrained_model = 'data/imagenet_models/VGG16.reduce.caffemodel'
snapshot_model = None
start_iter = 0
max_iters = 360000

cfg.DATA_DIR = '/home/workspace/datasets/VOC'
cfg.IMS_PER_BATCH = cfg.IMS_PER_BATCH / len(gpus)


if __name__ == '__main__':

    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)

    # setup caffe
    caffe.set_mode_gpu()

    # setup mpi
    if len(gpus) != mpi.Size():
        raise ValueError('Excepted {} mpi nodes, but got {}.'
                         .format(len(gpus), mpi.Size()))
    caffe.set_device(gpus[mpi.Rank()])
    mpi.Parallel([i for i in xrange(len(gpus))])
    mpi.Snapshot([0])
    if mpi.Rank() != 0:
        caffe.set_root_solver(False)

    # setup database
    cfg.DATABASE = imdb_name
    imdb = get_imdb(imdb_name)
    print 'Database({}): {} images will be used to train.'.format(cfg.DATABASE, imdb.db_size)
    output_dir = osp.abspath(osp.join(cfg.ROOT_DIR, 'output', cfg.EXP_DIR, args.imdb_name))
    print 'Output will be saved to `{:s}`'.format(output_dir)

    # train net
    train_net(solver_txt, output_dir,
              pretrained_model=pretrained_model,
              snapshot_model=snapshot_model,
              start_iter=start_iter,
              max_iters=max_iters)
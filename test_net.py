# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.vm.caffe as caffe
from datasets.factory import get_imdb
from core.test import test_net
from config import cfg
import time, os

cfg.DATA_DIR = '/home/workspace/datasets/VOC'
imdb_name = 'voc_2007_test'
gpu_id = 0
prototxt = 'models/pascal_voc/VGG16/test.prototxt'
caffemodel = 'output/voc_0712_trainval/SSD_VGG16_iter_70000.caffemodel'
vis = False

if __name__ == '__main__':

    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    net.name = os.path.splitext(os.path.basename(caffemodel))[0]
    imdb = get_imdb(imdb_name)

    test_net(net, imdb, thresh=cfg.TEST.THRESH, vis=vis)

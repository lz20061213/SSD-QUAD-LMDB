# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# Modified by Zhuang Liu
# --------------------------------------------------------

import os.path as osp

from core import make_db

if __name__ == '__main__':

    QUAD_ROOT_DIR = '/home/yangruyin/codes/SSD-QUAD-LMDB/data/quad'

    # train database: hand_train
    make_db(database_file=osp.join(QUAD_ROOT_DIR, 'cache/plane_2018_trainval_lmdb'),
            images_path=[osp.join(QUAD_ROOT_DIR, 'plane/JPEGImages')],
            annotations_path=[osp.join(QUAD_ROOT_DIR, 'plane/Annotations')],
            imagesets_path=[osp.join(QUAD_ROOT_DIR, 'plane/ImageSets/Main')],
            splits=['trainval'])

    # test database: hand_test
    make_db(database_file=osp.join(QUAD_ROOT_DIR, 'cache/plane_2018_test_lmdb'),
            images_path=osp.join(QUAD_ROOT_DIR, 'plane/JPEGImages'),
            annotations_path=osp.join(QUAD_ROOT_DIR, 'plane/Annotations'),
            imagesets_path=osp.join(QUAD_ROOT_DIR, 'plane/ImageSets/Main'),
            splits=['test'])


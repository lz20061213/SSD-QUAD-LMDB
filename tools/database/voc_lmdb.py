# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import os.path as osp

from core import make_db

if __name__ == '__main__':

    VOC_ROOT_DIR = '/home/workspace/datasets/VOC'

    # train database: voc_2007_trainval + voc_2012_trainval
    make_db(database_file=osp.join(VOC_ROOT_DIR, 'cache/voc_0712_trainval_lmdb'),
            images_path=[osp.join(VOC_ROOT_DIR, 'VOCdevkit2007/VOC2007/JPEGImages'),
                         osp.join(VOC_ROOT_DIR, 'VOCdevkit2012/VOC2012/JPEGImages')],
            annotations_path=[osp.join(VOC_ROOT_DIR, 'VOCdevkit2007/VOC2007/Annotations'),
                              osp.join(VOC_ROOT_DIR, 'VOCdevkit2012/VOC2012/Annotations')],
            imagesets_path=[osp.join(VOC_ROOT_DIR, 'VOCdevkit2007/VOC2007/ImageSets/Main'),
                            osp.join(VOC_ROOT_DIR, 'VOCdevkit2012/VOC2012/ImageSets/Main')],
            splits=['trainval', 'trainval'])

    # test database: voc_2007_test
    make_db(database_file=osp.join(VOC_ROOT_DIR, 'cache/voc_2007_test_lmdb'),
            images_path=osp.join(VOC_ROOT_DIR, 'VOCdevkit2007/VOC2007/JPEGImages'),
            annotations_path=osp.join(VOC_ROOT_DIR, 'VOCdevkit2007/VOC2007/Annotations'),
            imagesets_path=osp.join(VOC_ROOT_DIR, 'VOCdevkit2007/VOC2007/ImageSets/Main'),
            splits=['test'])
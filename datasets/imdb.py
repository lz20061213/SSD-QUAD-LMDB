# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import os
import os.path as osp

from core.io.data_reader import DataReader
from config import cfg

class imdb(object):
    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._data_path = ''
        self._image_ext = '.jpg'
        self._image_index = None
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def db_path(self):
        db_file = os.path.join(self.cache_path, self.name + '_lmdb')
        if not os.path.exists(db_file):
            print('except lmdb database @: {}, but is invalid.'.format(db_file))
            raise RuntimeError()
        return db_file

    @property
    def db_size(self):
        db_path = self.db_path
        data_reader = DataReader(**{'source': db_path})
        return data_reader.get_db_size()

    @property
    def num_images(self):
      return len(self.image_index)

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def _load_image_set_index(self):
        raise NotImplementedError

    def image_path_from_index(self, index):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        raise NotImplementedError

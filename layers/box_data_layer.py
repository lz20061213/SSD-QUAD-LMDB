# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.vm.caffe as caffe
import dragon.core.workspace as ws

from core.io.data_batch import DataBatch
from datasets.factory import get_imdb

from config import cfg


class BoxDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self._name_to_top_map = {}
        self._name_to_top_map['data'] = 0
        self._name_to_top_map['labels'] = 1
        self._name_to_top_map['im_info'] = 2
        self._imdb = get_imdb(cfg.DATABASE)
        self._data_batch = DataBatch(**{'source': self._imdb.db_path,
                                        'classes': self._imdb.classes,
                                        'shuffle': cfg.TRAIN.USE_SHUFFLE})

    def forward(self, bottom, top):
        blobs = self._data_batch.get()
        for blob_name, blob in blobs.items():
            top_ind = self._name_to_top_map[blob_name]
            ws.FeedTensor(top[top_ind], blob)
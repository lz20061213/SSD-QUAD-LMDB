# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# Modified by Zhuang Liu
# --------------------------------------------------------

from multiprocessing import Process
import numpy as np
import cv2
from six.moves import xrange

from dragon.config import logger
from config import cfg

from .utils import GetProperty


class BlobFetcher(Process):
    def __init__(self, **kwargs):
        super(BlobFetcher, self).__init__()
        self._partition = GetProperty(kwargs, 'partition', False)

        if self._partition:
            self._batch_size = int(self._batch_size / kwargs['group_size'])
        self.Q_in = self.Q_out = None
        self.daemon = True

        def cleanup():
            logger.info('Terminating BlobFetcher......')
            self.terminate()
            self.join()

        import atexit
        atexit.register(cleanup)

    def get_minibatch(self):
        num_images = cfg.TRAIN.IMS_PER_BATCH
        scale_h = cfg.TRAIN.RESIZE_PARAM['height']
        scale_w = cfg.TRAIN.RESIZE_PARAM['width']
        labels_list = []
        total_num_objs = 0

        # fill ims
        ims_blob = np.zeros(shape=(num_images, scale_h, scale_w, 3), dtype=np.float32)
        for i in xrange(num_images):
            im, labels = self.Q_in.get()
            #self.test_gt(im, labels, im.shape, winname='blob')
            ims_blob[i] = im.astype(np.float32, copy=False)
            ims_blob[i] -= cfg.PIXEL_MEANS
            labels = np.concatenate((np.array([i] * labels.shape[0]).reshape((-1, 1)), labels), axis=1)
            labels_list.append(labels)
            total_num_objs += labels.shape[0]

        ims_blob = ims_blob.transpose((0, 3, 1, 2))

        # fill labels
        # (total_num_objs, {item_id, norm_xmin, norm_ymin, norm_xmax, norm_ymax,
        # norm_x1, norm_y1, norm_x2, norm_y2,  norm_x3, norm_y3, norm_x4, norm_y4, cls})
        labels_blob = np.zeros(shape=(total_num_objs, 14), dtype=np.float32)
        start = 0
        for labels in labels_list:
            labels_blob[start: start + labels.shape[0]] = labels
            start += labels.shape[0]

        return {'data': ims_blob, 'labels': labels_blob,
                'im_info': np.array([scale_h, scale_w], dtype=np.float32)}

    def run(self):
        while True:
            self.Q_out.put(self.get_minibatch())

    def test_gt(self, im, boxes, im_shape, winname=None):
        # test the gt
        im = np.array(im)
        for i in range(boxes.shape[0]):
            width = im_shape[1]
            height = im_shape[0]
            box = boxes[i]
            xmin = int(box[0] * width)
            ymin = int(box[1] * height)
            xmax = int(box[2] * width)
            ymax = int(box[3] * height)
            x1 = box[4] * width
            y1 = box[5] * height
            x2 = box[6] * width
            y2 = box[7] * height
            x3 = box[8] * width
            y3 = box[9] * height
            x4 = box[10] * width
            y4 = box[11] * height
            assert xmin >=0 and xmax <= width-1 and ymin >= 0 and ymax <= height - 1, 'boundary error'
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1, 8)
            pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(im, [pts], True, (0, 255, 0), 1, 8)
        cv2.imshow(winname, im)
        cv2.waitKey(0)

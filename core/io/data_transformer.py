# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# Modified by Zhuang Liu
# --------------------------------------------------------

import numpy as np
import numpy.random as npr
from multiprocessing import Process
from six.moves import xrange

from dragon.config import logger

from config import cfg

from . import anno_pb2 as pb

from ..preprocessing import Distortor, Expander, Sampler, Resizer
from .utils import GetProperty, ReorderPoints

from ..utils.bbox_transform import clip_boxes

try:
    import cv2
except ImportError as e:
    pass


class DataTransformer(Process):
    def __init__(self, **kwargs):
        super(DataTransformer, self).__init__()
        self._distorter = Distortor(**cfg.TRAIN.DISTORT_PARAM)
        self._expander = Expander(**cfg.TRAIN.EXPAND_PARAM)
        self._sampler = Sampler(cfg.TRAIN.SAMPLERS)
        self._resizer = Resizer(**cfg.TRAIN.RESIZE_PARAM)
        self._random_seed = cfg.RNG_SEED
        self._mirror = cfg.TRAIN.USE_FLIPPED
        self._use_diff = cfg.TRAIN.USE_DIFF
        self._classes = GetProperty(kwargs, 'classes', ('__background__'))
        self._num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
        self._queues = []

        self.Q_in = self.Q_out = None
        self.daemon = True

        def cleanup():
            if not self.is_alive(): return
            logger.info('Terminating DataTransformer......')
            self.terminate()
            self.join()

        import atexit
        atexit.register(cleanup)

    # by lz.
    def _flip_boxes(self, boxes, width):
        _boxes = boxes.copy()
        for i in range(boxes.shape[0]):
            box = boxes[i]
            _box = _boxes[i]
            # xmin, xmax
            oldxmin = box[0].copy()
            oldxmax = box[2].copy()
            _box[0] = width - oldxmax - 1
            _box[2] = width - oldxmin - 1
            # points
            _points = [[width - box[4].copy() - 1, box[5]],
                       [width - box[6].copy() - 1, box[7]],
                       [width - box[8].copy() - 1, box[9]],
                       [width - box[10].copy() - 1, box[11]]]
            _points = ReorderPoints(_points)
            _box[4:] = sum(_points, [])
            assert (_box[2] >= _box[0]).all()
            #_boxes.append(_boxes)
        return _boxes

    def make_roidb(self, anno_datum, flip=False):
        annotations = anno_datum.annotation
        num_objs = 0
        if not self._use_diff:
            for anno in annotations:
                if not anno.difficult: num_objs += 1
        else:
            num_objs = len(annotations)

        # by lz.
        roidb = {
            'width': anno_datum.datum.width,
            'height': anno_datum.datum.height,
            'gt_classes': np.zeros((num_objs), dtype=np.int32),
            'boxes': np.zeros((num_objs, 12), dtype=np.float32),
            'normalized_boxes': np.zeros((num_objs, 12), dtype=np.float32),
        }

        ix = 0
        for anno in annotations:
            if not self._use_diff and anno.difficult: continue
            # by lz.
            p1_x = anno.x1
            p1_y = anno.y1
            p2_x = anno.x2
            p2_y = anno.y2
            p3_x = anno.x3
            p3_y = anno.y3
            p4_x = anno.x4
            p4_y = anno.y4
            points = [[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y], [p4_x, p4_y]]
            points = ReorderPoints(points)
            points = sum(points, [])
            xmin = np.min(points[0::2])
            ymin = np.min(points[1::2])
            xmax = np.max(points[0::2])
            ymax = np.max(points[1::2])
            roidb['boxes'][ix, :] = [xmin, ymin, xmax, ymax] + points
            roidb['gt_classes'][ix] = self._class_to_ind[anno.name]
            ix += 1

        if flip: roidb['boxes'] = self._flip_boxes(roidb['boxes'], roidb['width'])
        roidb['boxes'] = clip_boxes(roidb['boxes'], (roidb['height'], roidb['width']))
        roidb['normalized_boxes'][:, 0::2] = roidb['boxes'][:, 0::2] / float(roidb['width'])
        roidb['normalized_boxes'][:, 1::2] = roidb['boxes'][:, 1::2] / float(roidb['height'])
        return roidb

    def transform_image(self, serialized):
        anno_datum = pb.AnnotatedDatum()
        anno_datum.ParseFromString(serialized)
        datum = anno_datum.datum
        im = np.fromstring(datum.data, np.uint8)
        if datum.encoded is True:
            im = cv2.imdecode(im, -1)
        else:
            im = im.reshape((datum.height, datum.width, datum.channels))
        return im

    def transform_annos(self, serialized):
        anno_datum = pb.AnnotatedDatum()
        anno_datum.ParseFromString(serialized)
        filename = anno_datum.filename
        annotations = anno_datum.annotation
        objects = []
        for ix, anno in enumerate(annotations):
            obj_struct = {}
            obj_struct['name'] = anno.name
            obj_struct['difficult'] = int(anno.difficult)
            x1 = int(anno.x1)
            y1 = int(anno.y1)
            x2 = int(anno.x2)
            y2 = int(anno.y2)
            x3 = int(anno.x3)
            y3 = int(anno.y3)
            x4 = int(anno.x4)
            y4 = int(anno.y4)
            points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            points = ReorderPoints(points)
            points = sum(points, [])
            xmin = np.min(points[0::2])
            ymin = np.min(points[1::2])
            xmax = np.max(points[0::2])
            ymax = np.max(points[1::2])
            obj_struct['bbox'] = [xmin, ymin, xmax, ymax] + points
            objects.append(obj_struct)
        return filename, objects

    def transform_image_annos(self, serialized):
        anno_datum = pb.AnnotatedDatum()
        anno_datum.ParseFromString(serialized)
        datum = anno_datum.datum
        im = np.fromstring(datum.data, np.uint8)
        if datum.encoded is True:
            im = cv2.imdecode(im, -1)
        else:
            im = im.reshape((datum.height, datum.width, datum.channels))

        # do mirror ?
        flip = False
        if self._mirror:
            if npr.randint(0, 2) > 0:
                im = im[:, ::-1, :]
                flip = True

        # make roidb
        db = self.make_roidb(anno_datum, flip)
        boxes = db['normalized_boxes']
        gt_boxes = np.concatenate((boxes, db['gt_classes']
                                   .astype(np.float32).reshape((-1, 1)),), axis=1)

        #self.test_gt(im, gt_boxes, im.shape, winname='original')
        # distort => expand => sample => resize
        #
        im = self._distorter.distort_image(im)
        #self.test_gt(im, gt_boxes, im.shape, winname='distort')
        im, gt_boxes = self._expander.expand_image(im, gt_boxes)
        #self.test_gt(im, gt_boxes, im.shape, winname='expand')
        im, gt_boxes = self._sampler.sample_image(im, gt_boxes)
        #self.test_gt(im, gt_boxes, im.shape, winname='sample')
        im = self._resizer.resize_image(im)
        #self.test_gt(im, gt_boxes, im.shape, winname='resize')

        return [im, gt_boxes]

    def run(self):
        npr.seed(self._random_seed)
        while True:
            serialized = self.Q_in.get()
            datum = self.transform_image_annos(serialized)
            if len(datum[1]) < 1: continue
            self.Q_out.put(datum)

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
# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import os
from multiprocessing import Queue
try:
    import cPickle
except:
    import pickle as cPickle
from six.moves import xrange
import cv2
import numpy as np

import dragon.core.workspace as ws

from config import cfg, get_output_dir
from core.nms.nms_wrapper import nms
from core.io.data_batch import DataReader, DataTransformer
from core.utils.bbox_transform import bbox_transform_inv, clip_boxes
from core.utils.timer import Timer

def _get_image_blob(ims):
    scale_h = cfg.TRAIN.RESIZE_PARAM['height']
    scale_w = cfg.TRAIN.RESIZE_PARAM['width']
    processed_ims = []
    for im in ims:
        processed_ims.append(cv2.resize(im, (scale_w, scale_h)))
    processed_ims = np.array(processed_ims, dtype=np.float32)
    processed_ims -= cfg.PIXEL_MEANS
    im_blob = processed_ims.transpose((0, 3, 1, 2))
    #im_blob = np.array(processed_ims)
    return im_blob, np.array([scale_h, scale_w], dtype=np.float32)

def _get_blobs(ims):
    blobs = {'data' : None}
    blobs['data'], blobs['im_info'] = _get_image_blob(ims)
    return blobs

def _decode_boxes(prior_boxes, deltas, im_shape):
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
        deltas = deltas * np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS)
    boxes = bbox_transform_inv(prior_boxes, deltas)
    boxes[:, 0::2] *= im_shape[1]
    boxes[:, 1::2] *= im_shape[0]
    return clip_boxes(boxes, im_shape)

def im_detect(net, ims):
    blobs = _get_blobs(ims)
    forward_kwargs = {'data': blobs['data'], 'im_info': blobs['im_info']}
    net.forward(**forward_kwargs)()
    scores = ws.FetchTensor(net.blobs['mbox_prob'].data)
    prior_boxes = ws.FetchTensor(net.blobs['mbox_priorbox'].data)
    pred_deltas = ws.FetchTensor(net.blobs['mbox_loc_reshape'].data)
    pred_boxes = []
    for i in xrange(pred_deltas.shape[0]):
        pred_boxes.append(_decode_boxes(prior_boxes, pred_deltas[i], ims[i].shape))
    return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.5):
    inds = np.where(dets[:, -1] > thresh)[0]
    if len(inds) > 0:
        dets = dets[inds, :]
        im_copy = im.copy()
        for i in xrange(np.minimum(10, len(inds))):
            bbox = dets[i, :4]
            score = dets[i, -1]
            x1, y1, x2, y2 = bbox.astype(np.int32)[:]
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), (188,119,64), 2)
            #cv2.rectangle(im_copy, (x1 - 1, y1), (x2, y1 + 20), (188,119,64), -1)
            cv2.putText(im_copy, class_name + ' : %.2f' % score, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv2.imshow('SSD', im_copy)
        cv2.waitKey(0)

def vis_detections_all(im, class_name, dets, thresh=0.5):
    inds = np.where(dets[:, -1] > thresh)[0]
    if len(inds) > 0:
        dets = dets[inds, :]
        im_copy = im
        for i in xrange(np.minimum(100, len(inds))):
            bbox = dets[i, :4]
            poly = dets[i, 4:12]
            score = dets[i, -1]
            x1, y1, x2, y2 = bbox.astype(np.int32)[:]
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), (188,119,64), 1)
            cv2.rectangle(im_copy, (x1 - 1, y1), (x2, y1 + 20), (188,119,64), -1)
            cv2.putText(im_copy, class_name + ' : %.2f' % score, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            pts = np.array([[poly[0], poly[1]], [poly[2], poly[3]], [poly[4], poly[5]], [poly[6], poly[7]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(im, [pts], True, (0, 255, 0), 1, 8)

def test_net(net, imdb, thresh=0.1, vis=False):
    data_reader = DataReader(**{'source': imdb.db_path})
    data_reader.Q_out = Queue(cfg.TEST.BATCH_SIZE)
    data_reader.start()
    num_images = data_reader.get_db_size()
    data_transformer = DataTransformer(**{'classes': imdb.classes})
    num_classes = imdb.num_classes
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    imagenames = []
    recs = {}
    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for i in xrange(0, num_images, cfg.TEST.BATCH_SIZE):
        ims = []
        for j in xrange(cfg.TEST.BATCH_SIZE):
            if i + j >= num_images: continue
            serialized = data_reader.Q_out.get()
            im = data_transformer.transform_image(serialized)
            filename, objects = data_transformer.transform_annos(serialized)
            imagenames.append(filename)
            recs[filename] = objects
            ims.append(im)
        _t['im_detect'].tic()
        batch_scores, batch_boxes = im_detect(net, ims)
        _t['im_detect'].toc()
        _t['misc'].tic()
        for item_id in xrange(len(batch_scores)):
            global_id = i + item_id
            scores = batch_scores[item_id]
            boxes = batch_boxes[item_id]
            for j in xrange(1, num_classes):
                inds = np.where(scores[:, j] > thresh)[0]
                if len(inds) < 1: continue
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds]
                pre_nms_inds = np.argsort(-cls_scores)[0 : cfg.TEST.NMS_TOP_K]
                cls_scores = cls_scores[pre_nms_inds]
                cls_boxes = cls_boxes[pre_nms_inds]
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=True)
                cls_dets = cls_dets[keep, :]
                if vis: vis_detections_all(ims[item_id], imdb.classes[j], cls_dets, thresh)
                all_boxes[j][global_id] = cls_dets
            if vis:
                cv2.imshow('SSD', ims[item_id])
                cv2.waitKey(0)

            # Limit to max_per_image detections *over all classes*
            if cfg.TEST.MAX_PER_IMAGE > 0:
                image_scores = np.array([], dtype=np.float32)
                for j in xrange(1, imdb.num_classes):
                    if len(all_boxes[j][global_id]) < 1: continue
                    image_scores = np.hstack([image_scores, all_boxes[j][global_id][:, -1]])
                if len(image_scores) > cfg.TEST.MAX_PER_IMAGE:
                    image_thresh = np.sort(image_scores)[-cfg.TEST.MAX_PER_IMAGE]
                    for j in xrange(1, imdb.num_classes):
                        if len(all_boxes[j][global_id]) < 1: continue
                        keep = np.where(all_boxes[j][global_id][:, -1] >= image_thresh)[0]
                        all_boxes[j][global_id] = all_boxes[j][global_id][keep, :]

        _t['misc'].toc()
        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
                .format(i + 1, num_images, _t['im_detect'].average_time,
                        _t['misc'].average_time))

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir, imagenames, recs)
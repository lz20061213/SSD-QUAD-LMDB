# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.vm.caffe as caffe
from dragon.vm.caffe.model_libs import *
from dragon.vm.caffe import layers as L
from dragon.vm.caffe import params as P
import math
from backbone import AirBody


if __name__ == '__main__':

    net = caffe.NetSpec()
    net.data, net.gt_boxes, net.im_info = L.Python(ntop=3, module='layers.box_data_layer', layer='BoxDataLayer')
    AirBody(net, from_layer='data', use_conv5=False)
    mbox_source_layers = ['conv3b', 'conv4b']
    lr_mult = 1
    input_dim = 300
    min_sizes = [input_dim * 0.1, input_dim * 0.2]
    max_sizes = [input_dim * 0.3, input_dim * 0.5]
    aspect_ratios = [[2, 3], [2, 3]]
    steps = [8, 16]
    normalizations = [-1, -1]
    num_classes = 2
    mbox_layers = CreateMultiQuadHead(net, data_layer='data', from_layers=mbox_source_layers,
                                     use_batchnorm=False, min_sizes=min_sizes, max_sizes=max_sizes,
                                     aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
                                     num_classes=num_classes, flip=True, clip=False,
                                     kernel_size=3, pad=1, lr_mult=lr_mult)

    # Create the SoftmaxLayer
    name = "mbox_prob"
    softmax_inputs = [net.mbox_conf_reshape]
    net.mbox_prob = L.Softmax(*softmax_inputs, name=name, softmax_param={'axis': 2})

    with open('test.prototxt', 'w') as f:
        net_param = net.to_proto()
        del net_param.layer[0]
        net_param.input.extend(['data'])
        net_param.input_shape.extend([caffe_pb2.BlobShape(dim=[1, 3, input_dim, input_dim])])
        net_param.input.extend(['im_info'])
        net_param.input_shape.extend([caffe_pb2.BlobShape(dim=[2])])
        f.write(str(net_param))


    multibox_match_param = {
        'num_classes': num_classes,
        'overlap_threshold': 0.5,
        'ignore_cross_boundary_bbox': False,
    }
    # Create the MultiBoxMatchLayer.
    name = "mbox_match"
    match_inputs = [net.mbox_priorbox, net.gt_boxes]
    net.match_inds, net.match_labels, net.max_overlaps = L.Python(*match_inputs, name=name,
                module='layers.multibox_match_layer', layer='MultiBoxMatchLayer',
                param_str=str(multibox_match_param), ntop=3)


    # Create the HardMingLayer
    hard_mining_param = {
        'neg_pos_ratio': 3.0,
        'neg_overlap': 0.5,
    }
    name = "hard_mining"
    mining_inputs = [net.match_labels, net.max_overlaps, net.mbox_prob]
    net.labels = L.Python(*mining_inputs, name=name,
                          module='layers.hard_mining_layer', layer='HardMiningLayer',
                          param_str=str(hard_mining_param))

    # Create the LossLayer for cls
    name = "cls_loss"
    cls_loss_inputs = [net.mbox_conf_reshape, net.labels]
    net.cls_loss = L.SoftmaxWithLoss(*cls_loss_inputs, name=name,
            loss_param={'ignore_label': -1}, softmax_param={'axis': 2}, loss_weight=4.0)

    # Create the MultiBoxTargetLayer for bbox
    name = "mbox_target"
    bbox_target_inputs = [net.match_inds, net.labels,
                          net.mbox_priorbox, net.gt_boxes]
    net.bbox_targets, net.bbox_inside_weights, net.bbox_outside_weights = \
        L.Python(*bbox_target_inputs, name=name, ntop=3,
                 module='layers.multibox_target_layer', layer='MultiBoxTargetLayer')

    # Create the LossLayer for bbox
    name = "bbox_loss"
    bbox_loss_inputs = [net.mbox_loc_reshape, net.bbox_targets,
                        net.bbox_inside_weights, net.bbox_outside_weights]
    net.bbox_loss = L.SmoothL1Loss(*bbox_loss_inputs, name=name)

    with open('train.prototxt', 'w') as f:
        f.write(str(net.to_proto()))
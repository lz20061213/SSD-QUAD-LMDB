# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2018 SeetaTech
# Written by Zhuang Liu
# --------------------------------------------------------

import dragon.vm.caffe as caffe
from dragon.vm.caffe import params as P
from dragon.vm.caffe.model_libs import *
import math

def AddExtraLayers(net, use_batchnorm=True, lr_mult=1):
    use_relu = True

    # Add additional convolutional layers.
    # 32 x 32
    from_layer = net.keys()[-1]

    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    # 16 x 16
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2,
        lr_mult=lr_mult)

    # 8 x 8
    from_layer = out_layer
    out_layer = "conv7_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv7_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 4 x 4
    from_layer = out_layer
    out_layer = "conv8_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv8_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    # 2 x 2
    from_layer = out_layer
    out_layer = "conv9_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv9_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    # 1 x 1
    from_layer = out_layer
    out_layer = "conv10_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv10_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 4, 1, 1,
      lr_mult=lr_mult)

    return net

if __name__ == '__main__':

    imdb_name = 'voc_2007_trainval'
    net = caffe.NetSpec()
    net.data, net.gt_boxes, net.im_info = L.Python(ntop=3, module='layers.box_data_layer', layer='BoxDataLayer')
    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True, dropout=False)

    mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2', 'conv10_2']
    use_batchnorm = False
    lr_mult = 1
    min_sizes = []
    max_sizes = []
    min_ratio = 15
    max_ratio = 90
    step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
    min_dim = 512
    for ratio in xrange(min_ratio, max_ratio + 1, step):
        min_sizes.append(min_dim * ratio / 100.)
        max_sizes.append(min_dim * (ratio + step) / 100.)
    min_sizes = [min_dim * 7 / 100.] + min_sizes
    max_sizes = [min_dim * 15 / 100.] + max_sizes
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
    steps = [8, 16, 32, 64, 128, 256, 512]
    normalizations = [20, -1, -1, -1, -1, -1, -1]
    num_classes = 2
    share_location = True
    flip = True
    clip = False
    prior_variance = [0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult)
    mbox_layers = CreateMultiQuadHead(net, data_layer='data', from_layers=mbox_source_layers,
                                     use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
                                     aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
                                     num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
                                     prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)

    overlap_threshold = 0.5
    neg_pos_ratio = 3.
    neg_overlap = 0.5
    ignore_cross_boundary_bbox = False

    # Create the SoftmaxLayer
    name = "mbox_prob"
    softmax_inputs = [net.mbox_conf_reshape]
    net.mbox_prob = L.Softmax(*softmax_inputs, name=name, softmax_param={'axis': 2})

    with open('test.prototxt', 'w') as f:
        net_param = net.to_proto()
        del net_param.layer[0]
        net_param.input.extend(['data'])
        net_param.input_shape.extend([caffe_pb2.BlobShape(dim=[1, 3, min_dim, min_dim])])
        net_param.input.extend(['im_info'])
        net_param.input_shape.extend([caffe_pb2.BlobShape(dim=[2])])
        f.write(str(net_param))

    multibox_match_param = {
        'num_classes': num_classes,
        'overlap_threshold': overlap_threshold,
        'ignore_cross_boundary_bbox': ignore_cross_boundary_bbox,
    }

    # Create the MultiBoxMatchLayer.
    name = "mbox_match"
    match_inputs = [net.mbox_priorbox, net.gt_boxes]
    net.match_inds, net.match_labels, net.max_overlaps = L.Python(*match_inputs, name=name,
                module='layers.multibox_match_layer', layer='MultiBoxMatchLayer',
                param_str=str(multibox_match_param), ntop=3)


    # Create the HardMingLayer
    hard_mining_param = {
        'neg_pos_ratio': neg_pos_ratio,
        'neg_overlap': neg_overlap
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
            loss_param={'ignore_label': -1}, softmax_param={'axis': 2}, loss_weight=8.0)

    # Create the MultiBoxTargetLayer for bbox
    name = "mbox_target"
    bbox_target_inputs = [net.match_inds, net.labels,
                          net.mbox_priorbox, net.gt_boxes]
    net.bbox_targets, net.bbox_inside_weights, net.bbox_outside_weights = \
        L.Python(*bbox_target_inputs, name=name, ntop=3,
                 module='layers.multibox_target_layer', layer='MultiBoxTargetLayer')

    # Create the LossLayer for quad
    name = "bbox_loss"
    bbox_loss_inputs = [net.mbox_loc_reshape, net.bbox_targets,
                        net.bbox_inside_weights, net.bbox_outside_weights]
    net.bbox_loss = L.SmoothL1Loss(*bbox_loss_inputs, name=name)

    with open('train.prototxt', 'w') as f:
        f.write(str(net.to_proto()))
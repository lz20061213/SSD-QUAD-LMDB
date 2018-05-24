# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.vm.caffe import layers as L
from dragon.vm.caffe import params as P

def ResBlock(net, from_layer, out_layer, num_output, stride=1, force_branch1=False, use_memonger=True):
    ConvNormLayer(net, from_layer, out_layer + '_branch2a', 3, 1, stride, num_output, drop_bn=use_memonger)
    ConvNormLayer(net, out_layer + '_branch2a', out_layer + '_branch2b', 3, 1, 1, num_output,
                  no_relu=True, drop_conv=use_memonger, drop_bn=use_memonger)
    shortcut = from_layer
    if stride != 1 or force_branch1:
        ConvNormLayer(net, from_layer, out_layer + '_branch1', 1, 0, stride, num_output, no_relu=True)
        shortcut =  out_layer + '_branch1'
    net[out_layer] = L.Add(net[out_layer+'_branch2b'], net[shortcut])
    net[out_layer + '_relu'] = L.ReLU(net[out_layer], in_place=True)


def InceptionResBlock(net, from_layer, out_layer, num_output):
    ConvNormLayer(net, from_layer=from_layer, ks=1, p=0, s=1,
                  num_output=num_output, out_layer=out_layer + '_1x1')
    ConvNormLayer(net, from_layer=out_layer + '_1x1', ks=3, p=1, s=1,
                  num_output=num_output / 2, out_layer=out_layer + '_3x3_reduce')
    ConvNormLayer(net, from_layer=out_layer + '_3x3_reduce', ks=3, p=1, s=1,
                  num_output=num_output, out_layer=out_layer + '_3x3_a')
    ConvNormLayer(net, from_layer=out_layer + '_1x1', ks=3, p=1, s=1,
                  num_output=num_output, out_layer=out_layer + '_3x3_b')
    net[out_layer + '_concat'] = L.Concat(net[out_layer + '_1x1'],
                                       net[out_layer + '_3x3_a'],
                                       net[out_layer + '_3x3_b'], axis=1)
    ConvNormLayer(net, from_layer=out_layer + '_concat', ks=3, p=1, s=1,
                  num_output=num_output, out_layer=out_layer + '_reduce', no_relu=True)
    net[out_layer] = L.Add(net[out_layer + '_reduce'], net[from_layer])
    net[out_layer + '_relu'] = L.ReLU(net[out_layer], in_place=True)


bn_params = [{'lr_mult': 0, 'decay_mult': 0},
             {'lr_mult': 0, 'decay_mult': 0},
             {'lr_mult': 0, 'decay_mult': 0},
             {'lr_mult': 0, 'decay_mult': 0}]


def ConvNormLayer(net, from_layer, out_layer, ks, p, s, num_output,
                  no_relu=False, drop_conv=False, drop_bn=False):
    net[out_layer] = L.Convolution(net[from_layer], num_output=num_output, kernel_size=ks, stride=s, pad=p,
                                   weight_filler=dict(type='xavier'), bias_term=False, mirror_stage=drop_conv)
    net[out_layer + '_bn'] = L.BN(net[out_layer], in_place=True, mirror_stage=drop_bn,
                                  param=bn_params, batch_norm_param={'use_global_stats': True})
    if not no_relu:
        net[out_layer + '_relu'] = L.ReLU(net[out_layer + '_bn'], in_place=True)


def AirBody(net, from_layer='data', use_conv5=False):
    # conv1
    ConvNormLayer(net, from_layer, 'conv1', ks=7, p=3, s=2, num_output=64)
    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    # conv2
    ResBlock(net, 'pool1', 'conv2a', 64, force_branch1=True)
    ResBlock(net, 'conv2a', 'conv2b', 64)

    # conv3
    ResBlock(net, 'conv2b', 'conv3a', 128, stride=2)
    InceptionResBlock(net, 'conv3a', 'conv3b', 128)

    # conv4
    ResBlock(net, 'conv3b', 'conv4a', 256, stride=2)
    InceptionResBlock(net, 'conv4a', 'conv4b', 256)

    # conv5
    if use_conv5:
        ResBlock(net, 'conv4b', 'conv5a', 384, stride=2)
        InceptionResBlock(net, 'conv5a', 'conv5b', 384)



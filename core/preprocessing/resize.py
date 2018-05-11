# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import PIL.Image
import numpy.random as npr
import numpy as np

class Resizer(object):
    def __init__(self, **params):
        self._resize_height = params.get('height', 300)
        self._resize_width = params.get('width', 300)
        interp_list = {
            'LINEAR': PIL.Image.BILINEAR,
            'AREA': PIL.Image.BILINEAR,
            'NEAREST': PIL.Image.NEAREST,
            'CUBIC': PIL.Image.CUBIC,
            'LANCZOS4': PIL.Image.LANCZOS
        }
        interp_mode = params.get('interp_mode', None)
        if interp_mode is None: self._interp_mode = [PIL.Image.BILINEAR]
        else: self._interp_mode = [interp_list[key] for key in interp_mode]

    def resize_image(self, im):
        rand = npr.randint(0, len(self._interp_mode))
        im = PIL.Image.fromarray(im)
        im = im.resize((self._resize_width, self._resize_height), self._interp_mode[rand])
        return np.array(im)
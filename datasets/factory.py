# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

__sets = {}

from .pascal_voc import pascal_voc
from .detrac import detrac
from .quad import quad

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012', '0712']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for year in ['2017']:
    for split in ['train', 'test']:
        name = 'detrac_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: detrac(split, year))

for year in ['2018']:
    for split in ['trainval', 'test']:
        for classname in ['hand', 'plane']:
            name = '{}_{}_{}'.format(classname, year, split)
            __sets[name] = (lambda classname=classname, split=split, year=year: quad(classname, split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not name in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

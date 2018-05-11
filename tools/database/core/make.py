# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# Modified by Zhuang Liu
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import time
import cv2
import anno_pb2 as pb
from db import LMDB, wrapper_str

ZFILL = 8
ENCODE_QUALITY = 95

def set_zfill(value):
    global ZFILL
    ZFILL = value


def set_quality(value):
    global ENCODE_QUALITY
    ENCODE_QUALITY = value


def make_datum(image_file, xml_file):
    tree = ET.parse(xml_file)
    filename = os.path.split(xml_file)[-1]
    objs = tree.findall('object')

    anno_datum = pb.AnnotatedDatum()
    datum = pb.Datum()

    im = cv2.imread(image_file)
    datum.height, datum.width, datum.channels = im.shape
    datum.encoded = ENCODE_QUALITY != 100
    if datum.encoded:
        result, im = cv2.imencode('.jpg', im, [int(cv2.IMWRITE_JPEG_QUALITY), ENCODE_QUALITY])
    datum.data = im.tostring()
    anno_datum.datum.CopyFrom(datum)
    anno_datum.filename = filename.split('.')[0]

    for ix, obj in enumerate(objs):
        anno = pb.Annotation()
        bbox = obj.find('polygon') # by lz.
        x1 = float(bbox.find('x1').text)
        y1 = float(bbox.find('y1').text)
        x2 = float(bbox.find('x2').text)
        y2 = float(bbox.find('y2').text)
        x3 = float(bbox.find('x3').text)
        y3 = float(bbox.find('y3').text)
        x4 = float(bbox.find('x4').text)
        y4 = float(bbox.find('y4').text)
        cls = obj.find('name').text.lower().strip()
        anno.x1, anno.y1, anno.x2, anno.y2, anno.x3, anno.y3, anno.x4, anno.y4 = (x1, y1, x2, y2, x3, y3, x4, y4)
        anno.name = cls
        anno.difficult = False
        if obj.find('difficult') is not None:
            anno.difficult = int(obj.find('difficult').text) == 1
        anno_datum.annotation.add().CopyFrom(anno)

    return anno_datum


def make_db(database_file,
            images_path,
            annotations_path,
            imagesets_path,
            splits):
    if os.path.isdir(database_file) is True:
        raise ValueError('The database path is already exist.')
    else:
        root_dir = database_file[:database_file.rfind('/')]
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
    if not isinstance(images_path, list):
        images_path = [images_path]
    if not isinstance(annotations_path, list):
        annotations_path = [annotations_path]
    if not isinstance(imagesets_path, list):
        imagesets_path = [imagesets_path]
    assert len(splits) == len(imagesets_path)
    assert len(splits) == len(images_path)
    assert len(splits) == len(annotations_path)

    print('Start Time: ', time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))

    db = LMDB(max_commit=10000)
    db.open(database_file, mode='w')
    count = 0
    total_line = 0
    start_time = time.time()
    zfill_flag = '{0:0%d}' % (ZFILL)

    for db_idx, split in enumerate(splits):
        split_file = os.path.join(imagesets_path[db_idx], split + '.txt')
        assert os.path.exists(split_file)
        with open(split_file, 'r') as f:
            lines = f.readlines()
            total_line += len(lines)
        for line in lines:
            if count % 10000 == 0 and count > 0:
                now_time = time.time()
                print('{0} / {1} in {2:.2f} sec'.format(
                    count, total_line, now_time - start_time))
                db.commit()

            filename = line.strip()
            image_file = os.path.join(images_path[db_idx], filename + '.jpg')
            xml_file = os.path.join(annotations_path[db_idx], filename + '.xml')
            datum = make_datum(image_file, xml_file)
            if len(datum.annotation) == 0:
                print('Image({}) does not have annotations, ignore.'.format(filename + '.jpg'))
                continue
            count += 1
            db.put(zfill_flag.format(count - 1), datum.SerializeToString())

    now_time = time.time()
    print('{0} / {1} in {2:.2f} sec'.format(count, total_line, now_time - start_time))
    db.put('size', wrapper_str(str(count)))
    db.put('zfill', wrapper_str(str(ZFILL)))
    db.commit()
    db.close()
    end_time = time.time()

    print('{0} images have been stored in the database.'.format(total_line))
    print('This task finishes within {0:.2f} seconds.'.format(end_time - start_time))
    print('The size of database is {0} MB.'.format(
        float(os.path.getsize(database_file + '/data.mdb') / 1000 / 1000)))
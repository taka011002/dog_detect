import os
from pathlib import Path

import matplotlib.pyplot as plt

import chainer
import cv2

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import YOLOv3
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main(indir, outdir):
    flist = sorted(Path(indir).glob('*.jpg'))
    [recog(file, outdir) for file in flist]


def recog(file, outdir):
    model = YOLOv3(
        n_fg_class=len(voc_bbox_label_names),
        pretrained_model='voc0712')

    model.to_cpu()

    image = file
    img = utils.read_image(image, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    base_out_path = outdir + '/' + Path(file).stem
    for (b, l, s) in zip(bbox, label, score):
        dog_label_number = 11
        count = 1
        if l == dog_label_number:
            crop_image(str(file), base_out_path + '_' + str(count) + '.jpg', int(b[1]), int(b[0]), int(b[3]), int(b[2]))
            count += 1


def crop_image(infile, outfile, x, y, width, height):
    img = cv2.imread(infile)
    dst = img[y:height, x:width]
    cv2.imwrite(outfile, dst)


if __name__ == '__main__':
    main('chihuahua', 'croped_chihuahua')
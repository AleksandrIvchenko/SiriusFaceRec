import numpy as np
import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps
from IPython import embed

from mean_average_precision import MeanAveragePrecision

# [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
gt = np.array([
    [439, 157, 556, 241, 0, 0, 0],
    [437, 246, 518, 351, 0, 0, 0],
    [515, 306, 595, 375, 0, 0, 0],
    [407, 386, 531, 476, 0, 0, 0],
    [544, 419, 621, 476, 0, 0, 0],
    [609, 297, 636, 392, 0, 0, 0]
])

b = np.insert(gt, 4, 5, axis=1)

print (gt.shape)
print (gt)
print (b.shape)
print (b)

a_file = open("APTEST.txt", "w")
for row in b:
    np.savetxt(a_file, row)
a_file.close()

# [xmin, ymin, xmax, ymax, class_id, confidence]
preds = np.array([
    [429, 219, 528, 247, 0, 0.460851],
    [433, 260, 506, 336, 0, 0.269833],
    [518, 314, 603, 369, 0, 0.462608],
    [592, 310, 634, 388, 0, 0.298196],
    [403, 384, 517, 461, 0, 0.382881],
    [405, 429, 519, 470, 0, 0.369369],
    [433, 272, 499, 341, 0, 0.272826],
    [413, 390, 515, 459, 0, 0.619459]
])

# create metric_fn
metric_fn = MeanAveragePrecision(num_classes=1)

# add some samples to evaluation
for i in range(10):
    metric_fn.add(preds, gt)

#print ((metric_fn.__dict__))

# compute PASCAL VOC metric
print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")

# compute PASCAL VOC metric at the all points
print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")

# compute metric COCO metric
print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")

print (os.path.isdir('./ground_truth'))
#print (os.listdir())

#gt_dir = './ground_truth/'
#gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
#hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))

#print (gt_mat)
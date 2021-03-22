from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_re50
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
import time
import io

from itertools import product as product
from math import ceil

def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms

def cut_face(img, ldm, resize=256):
    dst = np.array([[38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.7299, 92.2041]], dtype=np.float32)
    dst = dst * resize / 112
    M, inliers = cv2.estimateAffinePartial2D(ldm, dst, method=cv2.RANSAC, ransacReprojThreshold=5)
    face = cv2.warpAffine(img, M, (resize, resize))
    return face


class PriorBox():
    def __init__(self, cfg, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def pipeline(img_raw, loc, conf, landms):
    confidence_threshold = 0.02
    top_k = 5000
    nms_threshold = 0.4
    keep_top_k = 750
    save_image = True
    vis_thres = 0.6

    torch.set_grad_enabled(False)
    cfg = cfg_re50

    device = 'cuda'
    resize = 1

    priors = PriorBox(cfg, image_size=(im_height, im_width)).forward()

    priors = priors.to(device)
    prior_data = priors.data

    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    # show image
    if save_image:
        for b in dets:
            if b[4] < vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))

    # save image
    landmarks = np.array([
        [b[5], b[6]],
        [b[7], b[8]],
        [b[9], b[10]],
        [b[11], b[12]],
        [b[13], b[14]]
    ], dtype=np.float32)

    face = cut_face(img=img_raw, ldm=landmarks, resize=256)
    name_rect_affin = 'toFE.jpg'
    cv2.imwrite(name_rect_affin, face)



    # Save landmarks to txt
    #file_name = image_path.split('/')[-1]
    landmarks_str = '{}  {}  {}  {}  {}  {}  {}  {}  {}  {}'.format(
        b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14])

    # saveing landmarks as additional features
    # text_file = open("landmarks.txt", "w")
    # text_file.write(landmarks_str)
    # text_file.close()

    return face, landmarks_str


if __name__ == '__main__':


    device = 'cuda'
    net = torch.jit.load("./weights/FaceDetector.pt", map_location=torch.device(device))
    net.eval()

    image_path = "./curve/scar.jpeg"

    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = net(img)  # forward pass

    pipeline(img_raw, loc, conf, landms)

    print("Sucess")

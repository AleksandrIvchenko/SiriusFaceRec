import cv2
import numpy as np
import sys
import torch
from itertools import product as product
from math import ceil
from PIL import Image


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def extractor_preprocessing(
        img,
        ldm,
        resize: int = 128,
    ):
    src = ldm
    dst = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    dst = dst * resize / 112
    M, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=5,
    )
    face = cv2.warpAffine(
        img,
        M,
        (resize, resize),
    )

    return face


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

    # print(priors.shape, loc.shape, priors.shape)
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
        self.image_size = 640, 640
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
    cfg = {'name': 'Resnet50', 'min_sizes': [[16, 32], [64, 128], [256, 512]], 'steps': [8, 16, 32], 'variance': [0.1, 0.2], 'clip': False, 'loc_weight': 2.0, 'gpu_train': True, 'batch_size': 24, 'ngpu': 4, 'epoch': 100, 'decay1': 70, 'decay2': 90, 'image_size': 840, 'pretrain': True, 'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3}, 'in_channel': 256, 'out_channel': 256}

    resize = 1

    img = expand2square(img_raw, (0, 0, 0))
    img.thumbnail((640, 640), Image.ANTIALIAS)
    img = np.array(img)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    priors = PriorBox(cfg, image_size=(im_height, im_width)).forward()

    loc = torch.from_numpy(loc)
    conf = torch.from_numpy(conf)
    landms = torch.from_numpy(landms)
    boxes = decode(loc.squeeze(0), priors, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.numpy()
    scores = conf.squeeze(0).numpy()[:, 1]

    landms = decode_landm(landms.squeeze(0), priors, cfg['variance'])
    scale1 = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                           img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                           img.shape[1], img.shape[0]])
    landms = landms * scale1 / resize
    landms = landms.numpy()

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
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    bb_squares = []
    if save_image:
        for b in dets:
            h = b[3] - b[1]
            w = b[2] - b[0]
            sq = h * w
            bb_squares.append(sq)
            if b[4] < vis_thres:
                continue
            b = list(map(int, b))


    b = list(map(int, dets[np.argmax(bb_squares)]))
    cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 4)
    cx = b[0]
    cy = b[1] + 12
    # Thith code can write a text
    # cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    landmarks = np.array([
        [b[5], b[6]],
        [b[7], b[8]],
        [b[9], b[10]],
        [b[11], b[12]],
        [b[13], b[14]]
    ], dtype=np.float32)

    face = cut_face(img=img, ldm=landmarks, resize=128)

    return face, landmarks, img_raw_bb


def detector_postprocessing(output0_data, output1_data, output2_data, raw_image):
    loc, conf, landms = output0_data, output1_data, output2_data
    image, landmarks, img_raw_bb = pipeline(raw_image, loc, conf, landms)

    return image, landmarks


def load_image(file):
    mean = np.array([0.485*255, 0.456*255, 0.406*255])
    std = np.array([0.229*255, 0.224*255, 0.225*255])
    size = 640, 640
    img = Image.open(file)
    img.load()
    img = img.convert('RGB')
    img = expand2square(img, (0, 0, 0))
    img.thumbnail(size, Image.ANTIALIAS)
    image_array = np.float32(img)
    image_array = (image_array - mean) / (std + sys.float_info.epsilon)
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = image_array[np.newaxis, ...]
    image_array = image_array.astype(np.float32)
    return image_array

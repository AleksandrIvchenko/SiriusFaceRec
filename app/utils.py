import sys

import cv2
import numpy as np
from PIL import Image


def extractor_preprocessing(
        img,
        ldm,
        resize: int = 128,
    ):
    src = np.array([
        [ldm[0], ldm[1]],
        [ldm[2], ldm[3]],
        [ldm[4], ldm[5]],
        [ldm[6], ldm[7]],
        [ldm[8], ldm[9]],
    ])
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


def load_image(file):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
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
    # resized_img = img.resize((640, 640), Image.BILINEAR)
    # resized_img = np.array(resized_img)
    # resized_img = resized_img.astype(np.float32)
    # resized_img = np.transpose(resized_img, (2, 0, 1))
    # resized_img = resized_img[np.newaxis, ...]
    return image_array


def detector_postprocessing(o1, o2, o3, image_array):
    pass

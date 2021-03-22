import cv2
import numpy as np


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
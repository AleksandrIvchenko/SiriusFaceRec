import cv2
import numpy as np


class LandmarksProcessor:
    @staticmethod
    def cut_face(
            img,
            box,
            ldm,
            resize: int = 256,
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

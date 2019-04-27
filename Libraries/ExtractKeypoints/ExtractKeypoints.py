import cv2
import numpy as np
from skimage.morphology import skeletonize

from Libraries.Preprocessing.Preprocessing import preprocessing


def noiseRemoval(img):
    tmp0 = np.array(img[:])
    tmp0 = np.array(tmp0)
    tmp1 = tmp0 / 255
    tmp2 = np.array(tmp1)
    tmp3 = np.array(tmp2)

    Img = np.array(tmp0)
    filTer = np.zeros((10, 10))
    W, H = tmp0.shape[:2]
    fsize = 6

    for i in range(W - fsize):
        for j in range(H - fsize):
            filTer = tmp1[i:i + fsize, j:j + fsize]

            flag = 0
            if sum(filTer[:, 0]) == 0:
                flag += 1
            if sum(filTer[:, fsize - 1]) == 0:
                flag += 1
            if sum(filTer[0, :]) == 0:
                flag += 1
            if sum(filTer[fsize - 1, :]) == 0:
                flag += 1
            if flag > 3:
                tmp2[i:i + fsize, j:j + fsize] = np.zeros((fsize, fsize))

    return tmp2


def extractKeypoints(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = preprocessing(img)
    img = np.array(img, dtype=np.uint8)

    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    img[img == 255] = 1

    skeleton = skeletonize(img)
    skeleton = np.array(skeleton, dtype=np.uint8)
    skeleton = noiseRemoval(skeleton)

    harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
    harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    threshold_harris = 125

    keypoints = []
    for x in range(0, harris_normalized.shape[0]):
        for y in range(0, harris_normalized.shape[1]):
            if harris_normalized[x][y] > threshold_harris:
                keypoints.append(cv2.KeyPoint(y, x, 1))

    orb = cv2.ORB_create()

    _, des = orb.compute(img, keypoints)
    return keypoints, des

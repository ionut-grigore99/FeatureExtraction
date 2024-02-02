import cv2
import numpy as np


def harris(img):
    return cv2.cornerHarris(img, 4, 3, 0.04)


def shi(img):
    detections = np.zeros_like(img, float)
    thresh = np.linspace(0.0001, 1, 100, endpoint=False)
    for t in thresh:
        corners = cv2.goodFeaturesToTrack(img, 100, t, 5)
        if corners is not None:
            corners = corners.astype(int)
            detections[(corners[:, 0, 1], corners[:, 0, 0])] = t
    return detections


def fast(img):
    detector = cv2.FastFeatureDetector_create(10)
    corners = detector.detect(img)
    detections = np.zeros_like(img, float)
    for c in corners:
        detections[tuple(np.flip(np.int0(c.pt),0))] = c.response
    return detections


def sift(img):
    sift = cv2.SIFT_create()
    corners, _ = sift.detectAndCompute(img, None)
    detections = np.zeros_like(img, float)
    for c in corners:
        detections[tuple(np.flip(np.int0(c.pt),0))] = c.response
    return detections


def disp(img):
    img = cv2.dilate(img, None)
    return img/(np.max(img) + 1e-7)


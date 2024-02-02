'''
According to the paper:
Mean Average Precision -> For each category, there are 1000 images sampled from the Synthetic Shapes generator.
Localization Error     -> For correct detections only, compute minimum distance between the detected keypoint and the ground truth.
'''

import sys
from itertools import cycle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches as mpatches
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from ..config.conf import LocalConf
from ..datasets.synthetic_dataset import FS, SyntheticDataset
from ..models.classical_detectors import disp, fast, harris, shi, sift
from ..pretrained.magicleap.superpoint import SuperPointNet
from ..pretrained.shaofengzeng.superpoint import SuperPointBNNet
from ..utils import box_nms, draw_mask


def extract_centroids(mask, connectivity=8):
    if not isinstance(mask, np.ndarray): mask = mask.squeeze().detach().numpy().copy()
    else: mask = mask.squeeze().copy()
    *_, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)

    return np.flip(centroids[1:].astype(int))  # exclude the background label (0)


def extract_predicted_keypoints(probs, conf_thresh):
    if not isinstance(probs, np.ndarray): probs = probs.squeeze().detach().numpy().copy()
    else: probs = probs.squeeze().copy()
    y, x = np.where(probs > conf_thresh)
    pos_points = list(zip(y, x))

    return pos_points, probs[y, x].tolist()


def is_point_in_patch(point, centroid, eps, shape):
    y, x   = point
    cy, cx = centroid
    h, w   = shape
    y_top   = max(0, cy - eps)
    y_bot   = min(h, cy + eps + 1)
    x_left  = max(0, cx - eps)
    x_right = min(w, cx + eps + 1)

    return y_top <= y < y_bot and x_left <= x < x_right


def compute_map_and_le(
        detector,
        weights_path,
        noise=True,
        eps=8,
        samples=1000,
        det_thresh=0.015,
        nms_det_thresh=0.001,
        nms_size=4,
        nms_topk=-1,
        plot=False
    ):

    conf = LocalConf().conf

    if detector == 'MagicPoint':
        sys.path.append(weights_path.parent.as_posix())
        from magic_point import MagicPoint #@NOTE: this is a hacky way to import the MagicPoint class

        net = MagicPoint(nms_det_thresh=nms_det_thresh, nms_size=nms_size, nms_topk=nms_topk)
        net.from_pretrained(weights_path=weights_path)

    elif detector == 'SuperPoint-shaofengzeng':
        net = SuperPointBNNet(det_thresh=nms_det_thresh, nms=nms_size, topk=nms_topk)
        net.from_pretrained(weights_path=weights_path)

    elif detector == 'SuperPoint-magicleap':
        net = SuperPointNet()
        net.from_pretrained(weights_path=weights_path)

    else: pass

    dataset = SyntheticDataset(im_sz=conf['im_sz'], noise=noise)
    y_gt, y_pred, y_pred_scores = [], [], []
    le = [] # localization error
    for i, f in enumerate(tqdm(cycle(FS))):
        im, mask = dataset.gen_shape_and_mask(f)

        if detector == 'MagicPoint':
            _, prob = net(im)

        elif detector == 'SuperPoint-shaofengzeng':
            output = net(im)
            prob   = output['det_info']['prob']

        elif detector == 'SuperPoint-magicleap':
            prob = im.squeeze().numpy()
            prob = box_nms(torch.from_numpy(net(prob)[None, ...]), min_prob=nms_det_thresh, size=nms_size, keep_top_k=nms_topk)

        elif detector=='Shi':
            img_shi = im.numpy().squeeze().astype(np.uint8) * 255
            prob    = disp(shi(img_shi))

        elif detector=='Harris':
            im_harris = im.numpy().squeeze().astype(np.uint8) * 255
            prob      = disp(harris(im_harris))

        elif detector=='SIFT':
            im_sift = im.numpy().squeeze().astype(np.uint8) * 255
            prob    = disp(sift(im_sift))

        elif detector=='FAST':
            im_fast = im.numpy().squeeze().astype(np.uint8) * 255
            prob    = disp(fast(im_fast))

        else:
            raise ValueError('Invalid detector name: {}'.format(detector))

        centroids    = extract_centroids(mask)
        kpts, scores = extract_predicted_keypoints(prob, conf_thresh=det_thresh)

        kpts_matched     = []
        kpts_not_matched = []
        kpts_scores      = []
        for k, s in zip(kpts, scores):
            if any([is_point_in_patch(k, c, eps=eps, shape=(mask.squeeze().shape)) for c in centroids]):
                kpts_matched.append(1)
                kpts_scores.append(s)
                # compute localization error only for correct detections
                le.append(np.linalg.norm(np.array(k) - np.array(centroids), axis=1).min())
                continue
            kpts_not_matched.append(1)
            kpts_scores.append(s)

        y_gt.extend([1] * len(kpts_matched) + [0] * len(kpts_not_matched))
        y_pred.extend(kpts_matched + kpts_not_matched)
        y_pred_scores.extend(kpts_scores)

        if plot:
            plt.imshow(draw_mask(im, mask.detach().numpy()))
            for k in kpts:
                plt.scatter(k[1], k[0], c='r', marker='+', s=25)
            red_patch   = mpatches.Patch(color='red', label='predicted')
            green_patch = mpatches.Patch(color='lime', label='ground truth')
            plt.legend(handles=[red_patch, green_patch], loc='upper right')
            plt.show()

        if i == samples:
            break

    print(f'{detector} @threshold: {nms_det_thresh}, @eps: {eps}, @noise: {noise} -> mAP: {average_precision_score(y_gt, y_pred_scores)}, localization error: {np.mean(le)}')

if __name__ == '__main__':
    SAMPLES        = 10 * 8 # 1_000 / category
    EPS            = 8
    DET_THRESH     = 0.015
    NMS_DET_THRESH = 0.015 #@NOTE: good rule of thumb to find out whether they trained with this threshold or not; MagicLeap performs well with 0.015 and poorly with 0.001 => they trained with a value closer to 0.015

    weights_dir = Path(__file__).parent.parent / 'pretrained'
    for NOISE in [True, False]:
        compute_map_and_le(detector='MagicPoint', weights_path=weights_dir / 'pytorch-training-2023-03-16-22-20-36-560/model.pth', noise=NOISE, eps=EPS, samples=SAMPLES, det_thresh=DET_THRESH, nms_det_thresh=NMS_DET_THRESH)
        compute_map_and_le(detector='MagicPoint', weights_path=weights_dir / 'pytorch-training-2023-04-12-17-40-59-097/model.pth', noise=NOISE, eps=EPS, samples=SAMPLES, det_thresh=DET_THRESH, nms_det_thresh=NMS_DET_THRESH)
        compute_map_and_le(detector='SuperPoint-shaofengzeng', weights_path=weights_dir / 'shaofengzeng/superpoint_bn.pth', noise=NOISE, eps=EPS, samples=SAMPLES, det_thresh=DET_THRESH, nms_det_thresh=NMS_DET_THRESH)
        compute_map_and_le(detector='SuperPoint-magicleap', weights_path=weights_dir / 'magicleap/superpoint_v1.pth', noise=NOISE, eps=EPS, samples=SAMPLES, det_thresh=DET_THRESH, nms_det_thresh=NMS_DET_THRESH)
        compute_map_and_le(detector='Shi', weights_path=None, noise=NOISE, eps=EPS, samples=SAMPLES, det_thresh=DET_THRESH, nms_det_thresh=NMS_DET_THRESH)
        compute_map_and_le(detector='Harris', weights_path=None, noise=NOISE, eps=EPS, samples=SAMPLES, det_thresh=DET_THRESH, nms_det_thresh=NMS_DET_THRESH)
        compute_map_and_le(detector='SIFT', weights_path=None, noise=NOISE, eps=EPS, samples=SAMPLES, det_thresh=DET_THRESH, nms_det_thresh=NMS_DET_THRESH)
        compute_map_and_le(detector='FAST', weights_path=None, noise=NOISE, eps=EPS, samples=SAMPLES, det_thresh=DET_THRESH, nms_det_thresh=NMS_DET_THRESH)

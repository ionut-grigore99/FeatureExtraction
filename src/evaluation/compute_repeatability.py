'''
Code adapted from: https://github.com/shaofengzeng/SuperPoint-Pytorch/blob/master/compute_repeatability.py
'''
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets.hpatches_dataset import HPatchesDataset
from ..models.classical_detectors import fast, harris, shi, sift
from ..pretrained.magicleap.superpoint import SuperPointNet
from ..pretrained.shaofengzeng.superpoint import SuperPointBNNet
from ..utils import box_nms


def draw_keypoints(img, corners, color=(0, 255, 0), radius=3, s=3): # no usage
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners).T:
        cv2.circle(img, tuple(s*np.flip(c, 0)), radius, color, thickness=-1)
    return img
#NO USAGE OF THIS FUNCTION


def select_top_k(prob, thresh=0, num=300):
    pts = np.where(prob > thresh)
    idx = np.argsort(prob[pts])[::-1][:num]
    pts = (pts[0][idx], pts[1][idx])
    return pts


def warp_keypoints(keypoints, H):
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],
                                        axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]


def filter_keypoints(points, shape):
    """ Keep only the points whose coordinates are
    inside the dimensions of shape. """
    mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) & \
           (points[:, 1] >= 0) & (points[:, 1] < shape[1])
    return points[mask, :]


def keep_true_keypoints(points, H, shape):
    """ Keep only the points whose warped coordinates by H
    are still inside shape. """
    warped_points = warp_keypoints(points[:, [1, 0]], H)
    warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
    mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) & \
           (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
    return points[mask, :]


def select_k_best(points, k):
    """ Select the k most probable points (and strip their proba).
    points has shape (num_points, 3) where the last coordinate is the proba. """
    sorted_prob = points[points[:, 2].argsort(), :2]
    start = min(k, points.shape[0])
    return sorted_prob[-start:, :]


def compute_repeatability(
        detector,
        weights_path,
        alteration,
        keep_k_points=300,
        eps=3,                  # threshold-ul din formula de Corr(xi) din appendixul din paper
        det_thresh=0.015,
        nms_det_thresh=0.001,   # pentru instantierea modelului MagicPoint
        nms_size=4,             # pentru instantierea modelului MagicPoint
        nms_topk=-1,            # pentru instantierea modelului MagicPoint
        plot=False
    ):

    aux_config = {
        'data':{
            'name': 'hpatches',
            'data_dir': '../hpatches/data/hpatches-sequences-release',
            'alteration': alteration,
            'use_cuda': torch.cuda.is_available(),
            'preprocessing': {
                'resize': [256, 160]
            },
        }
    }
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

    dataset_     = HPatchesDataset(aux_config['data'])
    p_dataloader = DataLoader(dataset_, batch_size=1, shuffle=True, collate_fn=dataset_.batch_collator)

    repeatability = []
    N1s, N2s      = [], []
    with torch.no_grad():
        for data in tqdm(p_dataloader):
            if detector == 'MagicPoint':
                _, prob1 = net(data['img'])
                _, prob2 = net(data['warp_img'])

            elif detector == 'SuperPoint-shaofengzeng':
                output1 = net(data['img'])
                output2 = net(data['warp_img'])
                prob1 = output1['det_info']['prob']
                prob2 = output2['det_info']['prob']

            elif detector == 'SuperPoint-magicleap':
                prob1 = data['img'].squeeze().numpy()
                prob2 = data['warp_img'].squeeze().numpy()
                prob1 = box_nms(torch.from_numpy(net(prob1)[None, ...]), min_prob=nms_det_thresh, size=nms_size, keep_top_k=nms_topk)
                prob2 = box_nms(torch.from_numpy(net(prob2)[None, ...]), min_prob=nms_det_thresh, size=nms_size, keep_top_k=nms_topk)

            elif detector=='Shi':
                img_shi = data['img'].numpy()
                img_shi = img_shi.squeeze()
                img_shi = img_shi.astype(np.float32) * 255

                img_shi_warped = data['warp_img'].numpy()
                img_shi_warped = img_shi_warped.squeeze()
                img_shi_warped = img_shi_warped.astype(np.float32) * 255
                prob1 = shi(img_shi)
                prob2 = shi(img_shi_warped)

            elif detector=='Harris':
                img_harris = data['img'].numpy()
                img_harris = img_harris.squeeze()
                img_harris = img_harris.astype(np.float32) * 255

                img_harris_warped = data['warp_img'].numpy()
                img_harris_warped = img_harris_warped.squeeze()
                img_harris_warped = img_harris_warped.astype(np.float32) * 255
                prob1 = harris(img_harris)
                prob2 = harris(img_harris_warped)

            elif detector=='SIFT':
                img_sift = data['img'].numpy()
                img_sift = img_sift.squeeze()
                img_sift = img_sift.astype(np.float32) * 255

                img_sift_warped = data['warp_img'].numpy()
                img_sift_warped = img_sift_warped.squeeze()
                img_sift_warped = img_sift_warped.astype(np.float32) * 255
                prob1 = sift(img_sift.astype(np.uint8))
                prob2 = sift(img_sift_warped.astype(np.uint8))

            elif detector=='FAST':
                img_fast = data['img'].numpy()
                img_fast = img_fast.squeeze()
                img_fast = img_fast.astype(np.float32) * 255

                img_fast_warped = data['warp_img'].numpy()
                img_fast_warped = img_fast_warped.squeeze()
                img_fast_warped = img_fast_warped.astype(np.float32) * 255
                prob1 = fast(img_fast.astype(np.uint8))
                prob2 = fast(img_fast_warped.astype(np.uint8))

            else:
                raise ValueError('Invalid detector name: {}'.format(detector))

            pred_rep = {'prob':prob1, 'warp_prob':prob2, 'homography': data['homography']}
            if not ('name' in data):
                pred_rep.update(data) #The line pred_rep.update(data) updates the pred_rep dictionary
                                      # by adding or updating key-value pairs from the data dictionary.
                                      #If any key in data already exists in pred_rep, its corresponding
                                      # value in pred_rep will be replaced with the value from data.
                                      #If a key in data is not present in pred_rep, it will be added to
                                      # pred_rep.

            if detector == 'MagicPoint' or detector == 'SuperPoint-shaofengzeng' or detector == 'SuperPoint-magicleap':
                pred_rep = {k:v.cpu().numpy().squeeze() for k, v in pred_rep.items()}
            else:
                pred_rep['homography']=pred_rep['homography'].cpu().numpy().squeeze()

            data = pred_rep #  the variable data will point to the same dictionary that pred_rep is pointing to
            shape = data['warp_prob'].shape
            H = data['homography']
            # basically pred_rep = {'prob':prob1, 'warp_prob':prob2, 'homography': data['homography']}

            ## Filter out predictions
            keypoints = np.where(data['prob'] > 0)
            prob = data['prob'][keypoints[0], keypoints[1]]
            keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
            warped_keypoints = np.where(data['warp_prob'] > 0)
            warped_prob = data['warp_prob'][warped_keypoints[0], warped_keypoints[1]]
            warped_keypoints = np.stack([warped_keypoints[0],
                                         warped_keypoints[1],
                                         warped_prob], axis=-1)

            warped_keypoints = keep_true_keypoints(warped_keypoints, np.linalg.inv(H), data['prob'].shape)

            # Warp the original keypoints with the true homography
            true_warped_keypoints = warp_keypoints(keypoints[:, [1, 0]], H)
            true_warped_keypoints = np.stack([true_warped_keypoints[:, 1],
                                              true_warped_keypoints[:, 0],
                                              prob], axis=-1)
            true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)

            # Keep only the keep_k_points best predictions
            warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
            true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)

            # Compute the repeatability
            N1 = true_warped_keypoints.shape[0]
            N2 = warped_keypoints.shape[0]
            N1s.append(N1)
            N2s.append(N2)
            true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
            warped_keypoints = np.expand_dims(warped_keypoints, 0)
            # shapes are broadcasted to N1 x N2 x 2:
            norm = np.linalg.norm(true_warped_keypoints - warped_keypoints, ord=None, axis=2)
            count1 = 0
            count2 = 0
            if N2 != 0:
                min1 = np.min(norm, axis=1)
                count1 = np.sum(min1 <= eps)
            if N1 != 0:
                min2 = np.min(norm, axis=0)
                count2 = np.sum(min2 <= eps)
            if N1 + N2 > 0:
                repeatability.append((count1 + count2) / (N1 + N2))

            if plot:
                d = pred_rep
                img = np.round(d['img'] * 255).astype(int).astype(np.uint8) #because of pred_rep.update(data)
                warp_img = np.round(d['warp_img'] * 255).astype(int).astype(np.uint8)

                points1 = select_top_k(d['prob'], thresh=det_thresh)
                points2 = select_top_k(d['warp_prob'], thresh=det_thresh)

                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(img, cmap='gray')
                for x, y in zip(*points1):
                    plt.scatter(y, x, s=25, c='red')
                plt.subplot(1, 2, 2)
                plt.imshow(warp_img, cmap='gray')
                for x, y in zip(*points2):
                    plt.scatter(y, x, s=25, c='red')
                plt.show()

    repeatability = np.mean(repeatability)
    print('@detector: {} | @alteration: {} |  @threshold: {} -> repeatability: {}'.format(detector, alteration, eps, repeatability))


if __name__=='__main__':

    weights_dir = Path(__file__).parent.parent / 'pretrained'

    for alteration in ['all', 'i', 'v']:
        compute_repeatability(detector='MagicPoint', weights_path=weights_dir / "pytorch_training_2023_03_16_22_20_36_560/model.pth", alteration=alteration, eps=1, det_thresh=0.015, nms_det_thresh=0.015, plot=False)
        compute_repeatability(detector='MagicPoint', weights_path=weights_dir / "pytorch_training_2023_03_16_22_20_36_560/model.pth", alteration=alteration, eps=3, det_thresh=0.015, nms_det_thresh=0.015, plot=False)
        compute_repeatability(detector='MagicPoint', weights_path=weights_dir / "pytorch_training_2023_03_16_22_20_36_560/model.pth", alteration=alteration, eps=8, det_thresh=0.015, nms_det_thresh=0.015, plot=False)

    for alteration in ['all', 'i', 'v']:
        compute_repeatability(detector='MagicPoint', weights_path=weights_dir / "pytorch_training_2023_04_12_17_40_59_097/model.pth", alteration=alteration, eps=1, det_thresh=0.015, nms_det_thresh=0.015, plot=False)
        compute_repeatability(detector='MagicPoint', weights_path=weights_dir / "pytorch_training_2023_04_12_17_40_59_097/model.pth", alteration=alteration, eps=3, det_thresh=0.015, nms_det_thresh=0.015, plot=False)
        compute_repeatability(detector='MagicPoint', weights_path=weights_dir / "pytorch_training_2023_04_12_17_40_59_097/model.pth", alteration=alteration, eps=8, det_thresh=0.015, nms_det_thresh=0.015, plot=False)

    for alteration in ['all', 'i', 'v']:
        compute_repeatability(detector='SuperPoint-shaofengzeng', weights_path=weights_dir / "shaofengzeng/superpoint_bn.pth", alteration=alteration, eps=1, det_thresh=0.015, nms_det_thresh=0.015, plot=False)
        compute_repeatability(detector='SuperPoint-shaofengzeng', weights_path=weights_dir / "shaofengzeng/superpoint_bn.pth", alteration=alteration, eps=3, det_thresh=0.015, nms_det_thresh=0.015, plot=False)
        compute_repeatability(detector='SuperPoint-shaofengzeng', weights_path=weights_dir / "shaofengzeng/superpoint_bn.pth", alteration=alteration, eps=8, det_thresh=0.015, nms_det_thresh=0.015, plot=False)

    for alteration in ['all', 'i', 'v']:
        compute_repeatability(detector='SuperPoint-magicleap', weights_path=weights_dir / "magicleap/superpoint_v1.pth", alteration=alteration, eps=1, det_thresh=0.015, nms_det_thresh=0.015, plot=False)
        compute_repeatability(detector='SuperPoint-magicleap', weights_path=weights_dir / "magicleap/superpoint_v1.pth", alteration=alteration, eps=3, det_thresh=0.015, nms_det_thresh=0.015, plot=False)
        compute_repeatability(detector='SuperPoint-magicleap', weights_path=weights_dir / "magicleap/superpoint_v1.pth", alteration=alteration, eps=8, det_thresh=0.015, nms_det_thresh=0.015, plot=False)

    # Classical Detectors
    # ===================

    for alteration in ['all', 'i', 'v']:
        compute_repeatability(detector='Shi', weights_path=None, alteration=alteration, keep_k_points=300, eps=1, det_thresh=1/65, plot=False)
        compute_repeatability(detector='Shi', weights_path=None, alteration=alteration, keep_k_points=300, eps=3, det_thresh=1/65, plot=False)
        compute_repeatability(detector='Shi', weights_path=None, alteration=alteration, keep_k_points=300, eps=8, det_thresh=1/65, plot=False)

    for alteration in ['all', 'i', 'v']:
        compute_repeatability(detector='Harris', weights_path=None, alteration=alteration, keep_k_points=300, eps=1, det_thresh=1/65, plot=False)
        compute_repeatability(detector='Harris', weights_path=None, alteration=alteration, keep_k_points=300, eps=3, det_thresh=1/65, plot=False)
        compute_repeatability(detector='Harris', weights_path=None, alteration=alteration, keep_k_points=300, eps=8, det_thresh=1/65, plot=False)

    for alteration in ['all', 'i', 'v']:
        compute_repeatability(detector='FAST', weights_path=None, alteration=alteration, keep_k_points=300, eps=1, det_thresh=1/65, plot=False)
        compute_repeatability(detector='FAST', weights_path=None, alteration=alteration, keep_k_points=300, eps=3, det_thresh=1/65, plot=False)
        compute_repeatability(detector='FAST', weights_path=None, alteration=alteration, keep_k_points=300, eps=8, det_thresh=1/65, plot=False)

    for alteration in ['all', 'i', 'v']:
        compute_repeatability(detector='SIFT', weights_path=None, alteration=alteration, keep_k_points=300, eps=1, det_thresh=1/65, plot=False)
        compute_repeatability(detector='SIFT', weights_path=None, alteration=alteration, keep_k_points=300, eps=3, det_thresh=1/65, plot=False)
        compute_repeatability(detector='SIFT', weights_path=None, alteration=alteration, keep_k_points=300, eps=8, det_thresh=1/65, plot=False)


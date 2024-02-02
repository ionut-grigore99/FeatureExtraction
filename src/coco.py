import os
import sys
from pathlib import Path

import numpy as np
import torch
from kornia.geometry.transform import warp_perspective
from lightglue import LightGlue, match_pair
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from ..config.conf import LocalConf
from ..utils import random_homography
from .augmentation import AlbumentationTransform


class Coco(Dataset):
    def __init__(self, root_dir, joint=False, max_num_keypoints=250, resolution=(640, 480)): # don't forget to change the resolution back to 640x480
        self.files = [Path(root_dir) / _ for _ in os.listdir(root_dir)]
        self.H, self.W = resolution
        self.max_num_keypoints = max_num_keypoints
        self.augment = AlbumentationTransform()
        self.joint = joint
        self.conf = LocalConf().conf
        self._use_label_smoothing = False
        self._label_smoothing_factor = 0.1
        self._use_spatially_awareness = False
        self._spatially_awareness_factor = 3

        self.load_extractor()
        self.load_matcher()

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        if self.joint:
            im = torch.from_numpy(np.array(Image.open(self.files[idx]).convert("L").resize((self.W, self.H))))[None, ...].float() / 255.
            H = random_homography(self.conf, key='joint_training')
            # if torch.cuda.is_available():
            #     im = im.cuda()
            #     H  = H.cuda()
            im_wp = warp_perspective(im[None, ...], H[None, ...], (self.H, self.W))[0]
            feats0, feats1, matches01 = match_pair(self.extractor, self.matcher, im, im_wp)
            kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
            m_kpts    = kpts0[matches[..., 0]].long().T
            m_kpts_wp = kpts1[matches[..., 1]].long().T
            # build masks
            mask    = self._get_mask(im.shape, m_kpts)
            mask_wp = self._get_mask(im_wp.shape, m_kpts_wp)
            return im.cpu(), mask.cpu(), im_wp.cpu(), mask_wp.cpu(), H.cpu()
        else:
            im = torch.from_numpy(np.array(Image.open(self.files[idx]).convert("L").resize((self.W, self.H))))[None, ...].float() / 255.
            # if torch.cuda.is_available(): im = im.cuda()
            #breakpoint()
            pred = self.extractor.extract(im)
            breakpoint()
            kpts = pred["keypoints"].squeeze().long().T
            im, mask = self._get_mask(im.shape, kpts, im)
            return im.cpu(), mask.cpu()

    def load_extractor(self):
        weights_dir = Path(__file__).parent.parent / "pretrained" / "magicleap"
        sys.path.append(weights_dir.as_posix())
        from inference import SuperPoint
        self.extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints)
        self.extractor.eval()
        # if torch.cuda.is_available(): self.extractor.cuda()

    def load_matcher(self):
        self.matcher = LightGlue(features="superpoint")
        self.matcher.eval()
        # if torch.cuda.is_available(): self.matcher.cuda()

    def _label_smoothing(self, mask, eps=0.1): return (1 - eps) * mask + eps * (1 - mask)

    def _rolling_window(self, lst, k): return [(lst[i:i+k]) for i in range(len(lst) - k + 1)]

    def _get_mask(self, shape, kpts, im=None):
        def _apply_mask_processing(mask):
            if self._use_spatially_awareness:
                mask = torch.from_numpy(gaussian_filter(mask.cpu().numpy(), sigma=self._spatially_awareness_factor))
                # renormalize
                mask = mask * (1 / mask.max())
            if self._use_label_smoothing:
                mask = self._label_smoothing(mask, eps=self._label_smoothing_factor)
            return mask

        mask = torch.zeros(shape)
        mask[:, kpts[1], kpts[0]] = 1.
        if self.joint:
            return _apply_mask_processing(mask)

        im, mask = self.augment(im, mask)
        mask = _apply_mask_processing(mask)
        return im, mask


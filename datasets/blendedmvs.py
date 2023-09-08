# -*- coding: utf-8 -*-
# @Description: Data preprocessing and organization for BlendedMVS dataset.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import os
import cv2
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms as T

from datasets.data_io import *


def motion_blur(img: np.ndarray, max_kernel_size=3):
    # Either vertial, hozirontal or diagonal blur
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_kernel_size + 1) / 2) * 2 + 1  # make sure is odd
    center = int((ksize - 1) / 2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid - center) + np.square(grid.T - center)) / (2. * var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    img = cv2.filter2D(img, -1, kernel)
    return img


class BlendedMVSDataset(Dataset):
    def __init__(self, root_dir, list_file, split, n_views, **kwargs):
        super(BlendedMVSDataset, self).__init__()

        self.levels = 4 
        self.root_dir = root_dir
        self.list_file = list_file
        self.split = split
        self.n_views = n_views

        assert self.split in ['train', 'val', 'all']

        self.scale_factors = {}
        self.scale_factor = 0

        self.img_wh = kwargs.get("img_wh", (768, 576))
        assert self.img_wh[0]%32==0 and self.img_wh[1]%32==0, \
            'img_wh must both be multiples of 2^5!'
        
        self.robust_train = kwargs.get("robust_train", True)
        self.augment = kwargs.get("augment", True)
        if self.augment:
            self.color_augment = T.ColorJitter(brightness=0.25, contrast=(0.3, 1.5))

        self.metas = self.build_metas()


    def build_metas(self):
        metas = []
        with open(self.list_file) as f:
            self.scans = [line.rstrip() for line in f.readlines()]
        for scan in self.scans:
            with open(os.path.join(self.root_dir, scan, "cams/pair.txt")) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) >= self.n_views-1:
                        metas += [(scan, ref_view, src_views)]
        return metas


    def read_cam_file(self, scan, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[-1])

        if scan not in self.scale_factors:
            self.scale_factors[scan] = 100.0 / depth_min
        depth_min *= self.scale_factors[scan]
        depth_max *= self.scale_factors[scan]
        extrinsics[:3, 3] *= self.scale_factors[scan]

        return intrinsics, extrinsics, depth_min, depth_max


    def read_depth_mask(self, scan, filename, depth_min, depth_max, scale):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth = (depth * self.scale_factors[scan]) * scale

        mask = (depth>=depth_min) & (depth<=depth_max)
        assert mask.sum() > 0
        mask = mask.astype(np.float32)
        if self.img_wh is not None:
            depth = cv2.resize(depth, self.img_wh, interpolation=cv2.INTER_NEAREST)
        h, w = depth.shape
        depth_ms = {}
        mask_ms = {}

        for i in range(self.levels):
            depth_cur = cv2.resize(depth, (w//(2**i), h//(2**i)), interpolation=cv2.INTER_NEAREST)
            mask_cur = cv2.resize(mask, (w//(2**i), h//(2**i)), interpolation=cv2.INTER_NEAREST)

            depth_ms[f"stage{self.levels-i}"] = depth_cur
            mask_ms[f"stage{self.levels-i}"] = mask_cur

        return depth_ms, mask_ms


    def read_img(self, filename):
        img = Image.open(filename)

        if self.augment:
            img = self.color_augment(img)
            img = motion_blur(np.array(img, dtype=np.float32))

        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img


    def __len__(self):
        return len(self.metas)


    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        
        if self.robust_train:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.n_views - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]
            scale_ratio = random.uniform(0.8, 1.25)
        else:
            view_ids = [ref_view] + src_views[:self.n_views - 1]
            scale_ratio = 1

        imgs = []
        mask = None
        depth = None
        depth_min = None
        depth_max = None

        proj={}
        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.root_dir, '{}/blended_images/{:0>8}.jpg'.format(scan, vid))
            depth_filename = os.path.join(self.root_dir, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.root_dir, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            img = self.read_img(img_filename)
            imgs.append(img.transpose(2,0,1))

            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(scan, proj_mat_filename)

            proj_mat_0 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_1 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_2 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_3 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            extrinsics[:3, 3] *= scale_ratio
            intrinsics[:2,:] *= 0.125
            proj_mat_0[0,:4,:4] = extrinsics.copy()
            proj_mat_0[1,:3,:3] = intrinsics.copy()
            int_mat_0 = intrinsics.copy()

            intrinsics[:2,:] *= 2
            proj_mat_1[0,:4,:4] = extrinsics.copy()
            proj_mat_1[1,:3,:3] = intrinsics.copy()
            int_mat_1 = intrinsics.copy()

            intrinsics[:2,:] *= 2
            proj_mat_2[0,:4,:4] = extrinsics.copy()
            proj_mat_2[1,:3,:3] = intrinsics.copy()
            int_mat_2 = intrinsics.copy()

            intrinsics[:2,:] *= 2
            proj_mat_3[0,:4,:4] = extrinsics.copy()
            proj_mat_3[1,:3,:3] = intrinsics.copy()
            int_mat_3 = intrinsics.copy()

            proj_matrices_0.append(proj_mat_0)
            proj_matrices_1.append(proj_mat_1)
            proj_matrices_2.append(proj_mat_2)
            proj_matrices_3.append(proj_mat_3)

            # reference view
            if i == 0:
                depth_min = depth_min_ * scale_ratio
                depth_max = depth_max_ * scale_ratio
                depth, mask = self.read_depth_mask(scan, depth_filename, depth_min, depth_max, scale_ratio)
                for l in range(self.levels):
                    mask[f'stage{l+1}'] = mask[f'stage{l+1}']
                    depth[f'stage{l+1}'] = depth[f'stage{l+1}']

        proj['stage1'] = np.stack(proj_matrices_0)
        proj['stage2'] = np.stack(proj_matrices_1)
        proj['stage3'] = np.stack(proj_matrices_2)
        proj['stage4'] = np.stack(proj_matrices_3)

        intrinsics_matrices = {
            "stage1": int_mat_0,
            "stage2": int_mat_1,
            "stage3": int_mat_2,
            "stage4": int_mat_3
        }
        
        sample = {
            "imgs": imgs,
            "proj_matrices": proj,
            "intrinsics_matrices": intrinsics_matrices,
            "depth": depth,
            "depth_values": np.array([depth_min, depth_max], dtype=np.float32),
            "mask": mask
        }

        return sample
# -*- coding: utf-8 -*-
# @Description: Data preprocessing and organization for Tanks and Temples dataset.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import os
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from datasets.data_io import *


class TNTDataset(Dataset):
    def __init__(self, root_dir, list_file, split, n_views, **kwargs):
        super(TNTDataset, self).__init__()

        self.root_dir = root_dir
        self.list_file = list_file
        self.split = split
        self.n_views = n_views

        self.cam_mode = kwargs.get("cam_mode", "origin")    # origin / short_range
        if self.cam_mode == 'short_range': assert self.split == "intermediate"
        self.img_mode = kwargs.get("img_mode", "resize")    # resize / crop

        self.total_depths = 192
        self.depth_interval_table = {
            # intermediate
            'Family': 2.5e-3, 'Francis': 1e-2, 'Horse': 1.5e-3, 'Lighthouse': 1.5e-2, 'M60': 5e-3, 'Panther': 5e-3, 'Playground': 7e-3, 'Train': 5e-3, 
            # advanced
            'Auditorium': 3e-2, 'Ballroom': 2e-2, 'Courtroom': 2e-2, 'Museum': 2e-2, 'Palace': 1e-2, 'Temple': 1e-2
        }
        self.img_wh = kwargs.get("img_wh", (-1, 1024))

        self.metas = self.build_metas()


    def build_metas(self):
        metas = []

        with open(os.path.join(self.list_file)) as f:
            scans = [line.rstrip() for line in f.readlines()]

        for scan in scans:
            with open(os.path.join(self.root_dir, self.split, scan, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) != 0:
                        metas += [(scan, -1, ref_view, src_views)]
        return metas

   
    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[-1])

        return intrinsics, extrinsics, depth_min, depth_max


    def read_img(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img


    def scale_tnt_input(self, intrinsics, img):
        if self.img_mode == "crop":
            intrinsics[1,2] = intrinsics[1,2] - 28  # 1080 -> 1024
            img = img[28:1080-28, :, :]
        elif self.img_mode == "resize": 
            height, width = img.shape[:2]

            max_w, max_h = self.img_wh[0], self.img_wh[1]
            if max_w == -1:
                max_w = width

            img = cv2.resize(img, (max_w, max_h))

            scale_w = 1.0 * max_w / width
            intrinsics[0, :] *= scale_w
            scale_h = 1.0 * max_h / height
            intrinsics[1, :] *= scale_h

        return intrinsics, img


    def __len__(self):
        return len(self.metas)


    def __getitem__(self, idx):
        scan, _, ref_view, src_views = self.metas[idx]
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        depth_min = None
        depth_max = None

        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.root_dir, self.split, scan, f'images/{vid:08d}.jpg')
            if self.cam_mode == 'short_range':
                # can only use for Intermediate
                proj_mat_filename = os.path.join(self.root_dir, self.split, scan, f'cams_{scan.lower()}/{vid:08d}_cam.txt')
            elif self.cam_mode == 'origin':
                proj_mat_filename = os.path.join(self.root_dir, self.split, scan, f'cams/{vid:08d}_cam.txt')

            img = self.read_img(img_filename)

            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(proj_mat_filename)
            intrinsics, img = self.scale_tnt_input(intrinsics, img)
            imgs.append(img.transpose(2,0,1))

            proj_mat_0 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_1 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_2 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_3 = np.zeros(shape=(2, 4, 4), dtype=np.float32)

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
                depth_min =  depth_min_
                if self.cam_mode == 'short_range':
                    depth_max = depth_min + self.total_depths * self.depth_interval_table[scan]
                elif self.cam_mode == 'origin':
                    depth_max = depth_max_

        proj={}
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
            "depth_values": np.array([depth_min, depth_max], dtype=np.float32),
            "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"
        }

        return sample
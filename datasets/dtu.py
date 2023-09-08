# -*- coding: utf-8 -*-
# @Description: Data preprocessing and organization for DTU dataset.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import os
import cv2
import random
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset

from datasets.data_io import *


class DTUDataset(Dataset):
    def __init__(self, root_dir, list_file, mode, n_views, **kwargs):
        super(DTUDataset, self).__init__()
        
        self.root_dir = root_dir
        self.list_file = list_file
        self.mode = mode
        self.n_views = n_views

        assert self.mode in ["train", "val", "test"]

        self.total_depths = 192
        self.interval_scale = 1.06

        self.data_scale = kwargs.get("data_scale", "mid")     # mid / raw
        self.robust_train = kwargs.get("robust_train", False)   # True / False
        self.color_augment = transforms.ColorJitter(brightness=0.5, contrast=0.5)

        if self.mode == "test":
            self.max_wh = kwargs.get("max_wh", (1600, 1200))

        self.metas = self.build_metas()

    
    def build_metas(self):
        metas = []

        with open(os.path.join(self.list_file)) as f:
            scans = [line.rstrip() for line in f.readlines()]

        pair_file = "Cameras/pair.txt"
        for scan in scans:
            with open(os.path.join(self.root_dir, pair_file)) as f:
                num_viewpoint = int(f.readline())

                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                    if self.mode == "train":
                        # light conditions 0-6
                        for light_idx in range(7):
                            metas.append((scan, light_idx, ref_view, src_views))
                    elif self.mode in ["test", "val"]:
                        if len(src_views) < self.n_views:
                            print("{} < num_views:{}".format(len(src_views), self.n_views))
                            src_views += [src_views[0]] * (self.n_views - len(src_views))
                        metas.append((scan, 3, ref_view, src_views))

        print("DTU Dataset in", self.mode, "mode metas:", len(metas))
        return metas


    def __len__(self):
        return len(self.metas)


    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        if self.mode == "test":
            intrinsics[:2, :] /= 4.0

        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])
        
        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.total_depths

        depth_interval *= self.interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval


    def read_img(self, filename):
        img = Image.open(filename)
        if self.mode == "train" and self.robust_train:
            img = self.color_augment(img)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img


    def crop_img(self, img):
        raw_h, raw_w = img.shape[:2]
        start_h = (raw_h-1024)//2
        start_w = (raw_w-1280)//2
        return img[start_h:start_h+1024, start_w:start_w+1280, :]  # (1024, 1280)

    
    def prepare_img(self, hr_img):
        h, w = hr_img.shape
        if self.data_scale == "mid":
            hr_img_ds = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
            h, w = hr_img_ds.shape
            target_h, target_w = 512, 640
            start_h, start_w = (h - target_h)//2, (w - target_w)//2
            hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]
        elif self.data_scale == "raw":
            hr_img_crop = hr_img[h//2-1024//2:h//2+1024//2, w//2-1280//2:w//2+1280//2]  # (1024, 1280)
        return hr_img_crop

    
    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=64):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics

    
    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage4": np_img,
        }
        return np_img_ms


    def read_depth_hr(self, filename, scale):
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32) * scale
        depth_lr = self.prepare_img(depth_hr)

        h, w = depth_lr.shape
        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(depth_lr, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage4": depth_lr,
        }
        return depth_lr_ms


    def __getitem__(self, idx):
        scan, light_idx, ref_view, src_views = self.metas[idx]

        if self.mode == "train" and self.robust_train:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.n_views-1)
            view_ids = [ref_view] + [src_views[i] for i in index]
            scale_ratio = random.uniform(0.8, 1.25) 
        else:
            view_ids = [ref_view] + src_views[:self.n_views-1]
            scale_ratio = 1

        imgs = []
        mask = None
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # @Note image & cam
            if self.mode in ["train", "val"]:
                if self.data_scale == "mid":
                    img_filename = os.path.join(self.root_dir, 'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid+1, light_idx))
                elif self.data_scale == "raw":
                    img_filename = os.path.join(self.root_dir, 'Rectified_raw/{}/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
                proj_mat_filename = os.path.join(self.root_dir, 'Cameras/train/{:0>8}_cam.txt').format(vid)
            elif self.mode == "test":
                img_filename = os.path.join(self.root_dir, 'Rectified/{}/rect_{:0>3}_3_r5000.png'.format(scan, vid+1))
                proj_mat_filename = os.path.join(self.root_dir, 'Cameras/{:0>8}_cam.txt'.format(vid))

            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            if self.mode in ["train", "val"]:
                if self.data_scale == "raw":
                    img = self.crop_img(img)
                    intrinsics[:2, :] *= 2.0
                if self.mode == "train" and self.robust_train:
                    extrinsics[:3,3] *= scale_ratio                    
            elif self.mode == "test":
                img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_wh[0], self.max_wh[1])

            imgs.append(img.transpose(2,0,1))

            # reference view
            if i == 0:
                # @Note depth values
                diff = 0.5 if self.mode in ["test", "val"] else 0
                depth_max = depth_interval * (self.total_depths - diff) + depth_min
                depth_values = np.array([depth_min * scale_ratio, depth_max * scale_ratio], dtype=np.float32)

                # @Note depth & mask
                if self.mode in ["train", "val"]:
                    depth_filename_hr = os.path.join(self.root_dir, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
                    depth = self.read_depth_hr(depth_filename_hr, scale_ratio)

                    mask_filename_hr = os.path.join(self.root_dir, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
                    mask = self.read_mask_hr(mask_filename_hr)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)
            
        proj_matrices = np.stack(proj_matrices)
        intrinsics = np.stack(intrinsics)
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2.0
        stage1_ins = intrinsics.copy()
        stage1_ins[:2, :] = intrinsics[:2, :] / 2.0
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_ins = intrinsics.copy()
        stage3_ins[:2, :] = intrinsics[:2, :] * 2.0
        stage4_pjmats = proj_matrices.copy()
        stage4_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
        stage4_ins = intrinsics.copy()
        stage4_ins[:2, :] = intrinsics[:2, :] * 4.0
        proj_matrices = {
            "stage1": stage1_pjmats,
            "stage2": proj_matrices,
            "stage3": stage3_pjmats,
            "stage4": stage4_pjmats
        }
        intrinsics_matrices = {
            "stage1": stage1_ins,
            "stage2": intrinsics,
            "stage3": stage3_ins,
            "stage4": stage4_ins
        }

        sample = {
            "imgs": imgs,
            "proj_matrices": proj_matrices,
            "intrinsics_matrices": intrinsics_matrices,
            "depth_values": depth_values
        }
        if self.mode in ["train", "val"]:
            sample["depth"] = depth
            sample["mask"] = mask
        elif self.mode == "test":
            sample["filename"] = scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"

        return sample
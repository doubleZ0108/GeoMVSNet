# -*- coding: utf-8 -*-
# @Description: Point cloud fusion strategy for Tanks and Temples dataset: DYnamic PCD.
#     Refer to: https://github.com/yhw-yhw/D2HC-RMVSNet/blob/master/fusion.py
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import os
import cv2
import signal
import numpy as np
from PIL import Image
from functools import partial
from multiprocessing import Pool
from plyfile import PlyData, PlyElement
import argparse
import re, json

from sklearn.preprocessing import scale

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="[/path/to/]tankandtemples/")
parser.add_argument('--out_dir', type=str, default='outputs/[exp_name]')
parser.add_argument('--ply_path', type=str, default='outputs/[exp_name]/dypcd_fusion_plys')

parser.add_argument('--split', type=str, default='intermediate', choices=['intermediate', 'advanced'])
parser.add_argument('--list_file', type=str, default='datasets/lists/tnt/intermediate.txt')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--single_processor', action='store_true')

parser.add_argument('--rescale', action='store_true')
parser.add_argument('--max_w', type=int)
parser.add_argument('--max_h', type=int)
parser.add_argument('--cam_mode', type=str, default='origin', choices=['origin', 'short_range'])
parser.add_argument('--img_mode', type=str, default='resize', choices=['resize', 'crop'])

parser.add_argument('--dist_base', type=float, default=1 / 4)
parser.add_argument('--rel_diff_base', type=float, default=1 / 1300)

args = parser.parse_args()


tnt_fusion_exps = [
    {
        "ply_path": "dypcd_fusion_plys_mean",
        "param_strategy": "mean",
    },
    {
        "ply_path": "dypcd_fusion_plys",
        "param_strategy": "hyper_param",
        "hyper_param_table": {    # -1 -> mean()
            'Family': 0.6,
            'Francis': 0.6,
            'Horse': 0.2,
            'Lighthouse': 0.7,
            'M60': 0.6,
            'Panther': 0.6,
            'Playground': 0.7,
            'Train': 0.6,

            'Auditorium': 0.1,
            'Ballroom': 0.4,
            'Courtroom': 0.4,
            'Museum': 0.5,
            'Palace': 0.5,
            'Temple': 0.4
        }
    },
]


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    K_xyz_reprojected[2:3][K_xyz_reprojected[2:3]==0] += 0.00001
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                 depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = None
    masks = []
    for i in range(2, 11):
        # mask = np.logical_and(dist < i / 4, relative_depth_diff < i / 1300)
        mask = np.logical_and(dist < i * args.dist_base, relative_depth_diff < i * args.rel_diff_base)
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src


def scale_input(intrinsics, img):
    if args.img_mode == "crop":
        intrinsics[1,2] = intrinsics[1,2] - 28  # 1080 -> 1024
        img = img[28:1080-28, :, :]
    elif args.img_mode == "resize": 
        height, width = img.shape[:2]
        img = cv2.resize(img, (width, 1024))
        scale_h = 1.0 * 1024 / height
        intrinsics[1, :] *= scale_h

    return intrinsics, img


def filter_depth(scene, root_dir, split, out_dir, plyfilename, fusion_exp):
    # num_stage = len(args.ndepths)

    # the pair file
    pair_file = os.path.join(root_dir, split, scene, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # src_views = src_views[:args.num_view]
        # load the camera parameters
        if args.cam_mode == 'short_range':
            ref_intrinsics, ref_extrinsics = read_camera_parameters(
                os.path.join(root_dir, split, scene, 'cams_{}/{:0>8}_cam.txt'.format(scene.lower(), ref_view)))
        elif args.cam_mode == 'origin':
            ref_intrinsics, ref_extrinsics = read_camera_parameters(
                os.path.join(root_dir, split, scene, 'cams/{:0>8}_cam.txt'.format(ref_view)))

        ref_img = read_img(os.path.join(root_dir, split, scene, 'images/{:0>8}.jpg'.format(ref_view)))
        ref_depth_est = read_pfm(os.path.join(out_dir, scene, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        confidence = read_pfm(os.path.join(out_dir, scene, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]

        if fusion_exp['param_strategy'] == 'mean':
            if ref_view % 50 == 0: print("-- thresh: {}".format(confidence.mean()))
            photo_mask = confidence > confidence.mean()
        elif fusion_exp['param_strategy'] == 'hyper_param':
            conf_thresh = fusion_exp['hyper_param_table'][scene]
            if conf_thresh == -1:
                photo_mask = confidence > confidence.mean()
                if ref_view % 50 == 0: print("-- thresh: mean() {}".format(confidence.mean()))
            else:
                photo_mask = confidence > conf_thresh
                if ref_view % 50 == 0: print("-- thresh: {}".format(conf_thresh))
            
        
        flag_img = ref_img
        ref_intrinsics, _ = scale_input(ref_intrinsics, flag_img)

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        dy_range = len(src_views) + 1
        geo_mask_sums = [0] * (dy_range - 2)
        for src_view in src_views:
            # camera parameters of the source view
            if args.cam_mode == 'short_range':
                src_intrinsics, src_extrinsics = read_camera_parameters(
                    os.path.join(root_dir, split, scene, 'cams_{}/{:0>8}_cam.txt'.format(scene.lower(), src_view)))
            elif args.cam_mode == 'origin':
                src_intrinsics, src_extrinsics = read_camera_parameters(
                    os.path.join(root_dir, split, scene, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_dir, scene, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            src_intrinsics, _ = scale_input(src_intrinsics, flag_img)
                
            masks, geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics,
                                                                                               ref_extrinsics, src_depth_est,
                                                                                               src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            for i in range(2, dy_range):
                geo_mask_sums[i - 2] += masks[i - 2].astype(np.int32)

            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least args.thres_view source views matched
        geo_mask = geo_mask_sum >= dy_range
        for i in range(2, dy_range):
            geo_mask = np.logical_or(geo_mask, geo_mask_sums[i - 2] >= i)

        final_mask = np.logical_and(photo_mask, geo_mask)

        if ref_view < 3:
            os.makedirs(os.path.join(out_dir, scene, "mask"), exist_ok=True)
            save_mask(os.path.join(out_dir, scene, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
            save_mask(os.path.join(out_dir, scene, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
            save_mask(os.path.join(out_dir, scene, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{:.3f}/{:.3f}/{:.3f}".format(os.path.join(out_dir, scene), ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))
        
        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        print("valid_points {:.3f}".format(valid_points.mean()))
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
 
        # color = ref_img[:-24, :, :][valid_points]
        color = ref_img[28:1080-28, :, :][valid_points]

        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def dypcd_filter_worker(scene):
    save_name = '{}.ply'.format(scene)

    filter_depth(scene, args.root_dir, args.split, args.out_dir, os.path.join(args.out_dir, fusion_exp['ply_path'], save_name), fusion_exp)


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == '__main__':
    
    with open(os.path.join(args.list_file)) as f:
        testlist = [line.rstrip() for line in f.readlines()]

    for fusion_exp in tnt_fusion_exps:

        if not os.path.isdir(os.path.join(args.out_dir, fusion_exp['ply_path'])):
            os.mkdir(os.path.join(args.out_dir, fusion_exp['ply_path']))
        

        if args.single_processor:
            for scene in testlist:
                save_name = '{}.ply'.format(scene)
                filter_depth(scene, args.root_dir, args.split, args.out_dir, os.path.join(args.out_dir, fusion_exp['ply_path'], save_name), fusion_exp)

        else:
            partial_func = partial(dypcd_filter_worker)
            p = Pool(args.num_workers, init_worker)
            try:
                p.map(partial_func, testlist)
            except KeyboardInterrupt:
                print("....\nCaught KeyboardInterrupt, terminating workers")
                p.terminate()
            else:
                p.close()
            p.join()
# -*- coding: utf-8 -*-
# @Description: Main process of network training & evaluation.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import os, sys, time, gc, datetime, logging, json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from datasets.dtu import DTUDataset
from datasets.blendedmvs import BlendedMVSDataset

from models.geomvsnet import GeoMVSNet
from models.loss import geomvsnet_loss
from models.utils import *
from models.utils.opts import get_opts


cudnn.benchmark = True
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1

args = get_opts()


def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    if args.lr_scheduler == 'MS':
        milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
        lr_gamma = 1 / float(args.lrepochs.split(':')[1])
        lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=500, last_epoch=len(TrainImgLoader) * start_epoch - 1)
    elif args.lr_scheduler == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs*len(TrainImgLoader)), eta_min=0)
    elif args.lr_scheduler == 'onecycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=int(args.epochs*len(TrainImgLoader)))
    elif args.lr_scheduler == 'lambda':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** ((epoch-1) / len(TrainImgLoader)), last_epoch=len(TrainImgLoader)*start_epoch-1)


    for epoch_idx in range(start_epoch, args.epochs):
        logger.info('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample, args)
            lr_scheduler.step()
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    if not args.notensorboard:
                        tb_save_scalars(tb_writer, 'train', scalar_outputs, global_step)
                        tb_save_images(tb_writer, 'train', image_outputs, global_step)
                    logger.info("Epoch {}/{}, Iter {}/{}, 2mm_err={:.3f} | lr={:.6f}, train_loss={:.3f}, abs_err={:.3f}, pw_loss={:.3f}, dds_loss={:.3f}, time={:.3f}".format(
                           epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                           scalar_outputs["thres2mm_error"],
                           optimizer.param_groups[0]["lr"], 
                           loss,
                           scalar_outputs["abs_depth_error"],
                           scalar_outputs["s3_pw_loss"],
                           scalar_outputs["s3_dds_loss"],
                           time.time() - start_time))
                del scalar_outputs, image_outputs

        # save checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            if ((epoch_idx + 1) % args.save_freq == 0) or (epoch_idx == args.epochs-1):
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/model_{:0>2}.ckpt".format(args.logdir, epoch_idx))  
        gc.collect()

        # testing
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        if not args.notensorboard:
                            tb_save_scalars(tb_writer, 'test', scalar_outputs, global_step)
                            tb_save_images(tb_writer, 'test', image_outputs, global_step)
                        logger.info(
                            "Epoch {}/{}, Iter {}/{}, 2mm_err={:.3f} | lr={:.6f}, test_loss={:.3f}, abs_err={:.3f}, pw_loss={:.3f}, dds_loss={:.3f}, time={:.3f}".format(
                            epoch_idx, args.epochs, batch_idx, len(TestImgLoader),
                            scalar_outputs["thres2mm_error"],
                            optimizer.param_groups[0]["lr"], 
                            loss,
                            scalar_outputs["abs_depth_error"],
                            scalar_outputs["s3_pw_loss"],
                            scalar_outputs["s3_dds_loss"],
                            time.time() - start_time))
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs, image_outputs

            if (not is_distributed) or (dist.get_rank() == 0):
                if not args.notensorboard:
                    tb_save_scalars(tb_writer, 'fulltest', avg_test_scalars.mean(), global_step)
                logger.info("avg_test_scalars: " + json.dumps(avg_test_scalars.mean()))
            gc.collect()


def train_sample(model, model_loss, optimizer, sample, args):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt_ms, mask_ms = sample_cuda["depth"], sample_cuda["mask"]
    depth_gt, mask = depth_gt_ms["stage{}".format(args.levels)], mask_ms["stage{}".format(args.levels)]

    # @Note GeoMVSNet main
    outputs = model(
        sample_cuda["imgs"], 
        sample_cuda["proj_matrices"], sample_cuda["intrinsics_matrices"], 
        sample_cuda["depth_values"]
    )

    depth_est = outputs["depth"]

    loss, epe, pw_loss_stages, dds_loss_stages = model_loss(
        outputs, depth_gt_ms, mask_ms, 
        stage_lw=[float(e) for e in args.stage_lw.split(",") if e], depth_values=sample_cuda["depth_values"]
    )

    loss.backward()
    optimizer.step()

    scalar_outputs = {
        "loss": loss,
        "epe": epe,
        "s0_pw_loss": pw_loss_stages[0],
        "s1_pw_loss": pw_loss_stages[1],
        "s2_pw_loss": pw_loss_stages[2],
        "s3_pw_loss": pw_loss_stages[3],
        "s0_dds_loss": dds_loss_stages[0],
        "s1_dds_loss": dds_loss_stages[1],
        "s2_dds_loss": dds_loss_stages[2],
        "s3_dds_loss": dds_loss_stages[3],
        "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
        "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
        "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
        "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
    }

    image_outputs = {
        "depth_est": depth_est * mask,
        "depth_est_nomask": depth_est,
        "depth_gt": sample["depth"]["stage1"],
        "ref_img": sample["imgs"][0],
        "mask": sample["mask"]["stage1"],
        "errormap": (depth_est - depth_gt).abs() * mask,
    }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


@make_nograd_func
def test_sample_depth(model, model_loss, sample, args):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms, mask_ms = sample_cuda["depth"], sample_cuda["mask"]
    depth_gt, mask = depth_gt_ms["stage{}".format(args.levels)], mask_ms["stage{}".format(args.levels)]

    outputs = model_eval(
        sample_cuda["imgs"], 
        sample_cuda["proj_matrices"], sample_cuda["intrinsics_matrices"], 
        sample_cuda["depth_values"]
    )
    
    depth_est = outputs["depth"]

    loss, epe, pw_loss_stages, dds_loss_stages = model_loss(
        outputs, depth_gt_ms, mask_ms, 
        stage_lw=[float(e) for e in args.stage_lw.split(",") if e], depth_values=sample_cuda["depth_values"]
    )
    
    scalar_outputs = {
        "loss": loss,
        "epe": epe,
        "s0_pw_loss": pw_loss_stages[0],
        "s1_pw_loss": pw_loss_stages[1],
        "s2_pw_loss": pw_loss_stages[2],
        "s3_pw_loss": pw_loss_stages[3],
        "s0_dds_loss": dds_loss_stages[0],
        "s1_dds_loss": dds_loss_stages[1],
        "s2_dds_loss": dds_loss_stages[2],
        "s3_dds_loss": dds_loss_stages[3],
        "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
        "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
        "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
        "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
    }

    image_outputs = {
        "depth_est": depth_est * mask,
        "depth_est_nomask": depth_est,
        "depth_gt": sample["depth"]["stage1"],
        "ref_img": sample["imgs"][0],
        "mask": sample["mask"]["stage1"],
        "errormap": (depth_est - depth_gt).abs() * mask
    }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


def initLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
    logfile = os.path.join(args.logdir, 'train-' + curTime + '.log')
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fileHandler = logging.FileHandler(logfile, mode='a')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.info("Logger initialized.")
    logger.info("Writing logs to file: {}".format(logfile))
    logger.info("Current time: {}".format(curTime))

    settings_str = "All settings:\n"
    for k,v in vars(args).items(): 
        settings_str += '{0}: {1}\n'.format(k,v)
    logger.info(settings_str)

    return logger


if __name__ == '__main__':
    logger = initLogger()

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    set_random_seed(args.seed)
    device = torch.device(args.device)


    # tensorboard
    if (not is_distributed) or (dist.get_rank() == 0):
        if not os.path.isdir(args.logdir):
            os.makedirs(args.logdir)
        current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        logger.info("current time " + current_time_str)
        logger.info("creating new summary file")
        if not args.notensorboard:
            tb_writer = SummaryWriter(args.logdir)


    # @Note GeoMVSNet model
    model = GeoMVSNet(
        levels=args.levels, 
        hypo_plane_num_stages=[int(n) for n in args.hypo_plane_num_stages.split(",")], 
        depth_interal_ratio_stages=[float(ir) for ir in args.depth_interal_ratio_stages.split(",")],
        feat_base_channel=args.feat_base_channel, 
        reg_base_channel=args.reg_base_channel,
        group_cor_dim_stages=[int(n) for n in args.group_cor_dim_stages.split(",")],
    )
    model.to(device)

    model_loss = geomvsnet_loss

    # optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)


    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        logger.info("resuming: " + loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1


    # distributed
    if (not is_distributed) or (dist.get_rank() == 0):
        logger.info("start at epoch {}".format(start_epoch))
        logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if is_distributed:
        if dist.get_rank() == 0:
            logger.info("Let's use {} GPUs in distributed mode!".format(torch.cuda.device_count()))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True,
        )
    else:
        if torch.cuda.is_available():
            logger.info("Let's use {} GPUs in parallel mode.".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)


    # dataset, dataloader
    if args.which_dataset == "dtu":
        train_dataset = DTUDataset(args.trainpath, args.trainlist, "train", args.n_views, data_scale=args.data_scale, robust_train=args.robust_train)
        test_dataset = DTUDataset(args.testpath, args.testlist, "val", args.n_views, data_scale=args.data_scale)
    elif args.which_dataset == "blendedmvs":
        train_dataset = BlendedMVSDataset(args.trainpath, args.trainlist, "train", args.n_views, img_wh=(768, 576), robust_train=args.robust_train, augment=False)
        test_dataset = BlendedMVSDataset(args.testpath, args.testlist, "val", args.n_views, img_wh=(768, 576))

    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())

        TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=8, drop_last=True, pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, num_workers=8, drop_last=False, pin_memory=args.pin_m)
    else:
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=args.pin_m)

    train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)
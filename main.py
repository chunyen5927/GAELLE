%matplotlib inline

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.cuda.amp import autocast as autocast
from apex.parallel.LARC import LARC

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    shuffle_logits,
    concat_local_logits,
)
from src.multicropdataset import MultiCropDataset
import src.resnet50 as resnet_models

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of SwAV")

parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--local_feat_dim", default=128, type=int,
                    help="local feature dimension")
parser.add_argument("--nmb_prototypes", default=3000, type=int,
                    help="number of prototypes")
parser.add_argument("--nmb_local_prototypes", default=3000, type=int,
                    help="number of local prototypes")
parser.add_argument("--nmb_groups_heads", default=8, type=int, help="heads of channels grouping")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--local_queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")

parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")
parser.add_argument("--alpha", default=0.5, type=float, help="loss weight")

parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")


def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    train_dataset = MultiCropDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        local_output_dim=args.local_feat_dim,
        nmb_prototypes=args.nmb_prototypes,
        nmb_local_prototypes=args.nmb_local_prototypes,
        nmb_groups_heads = args.nmb_groups_heads
    )

    # synchronize batch norm layers
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on]
    )
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # GradScaler to support autocast for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    if args.use_fp16:
        logger.info("Building mixed precision scaler done.")

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        scaler=scaler
    )
    start_epoch = to_restore["epoch"]

    # build the queue
    queue = None
    queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

    local_queue = None
    local_queue_path = os.path.join(args.dump_path, "local_queue" + str(args.rank) + ".pth")
    if os.path.isfile(local_queue_path):
        local_queue = torch.load(local_queue_path)["local_queue"]
    args.local_queue_length -= args.local_queue_length % ((args.local_feat_dim // args.nmb_groups_heads) * args.batch_size * args.world_size)

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()

        if args.local_queue_length > 0 and epoch >= args.epoch_queue_starts and local_queue is None:
            local_queue = torch.zeros(
                len(args.crops_for_assign),
                args.local_queue_length // args.world_size,
                args.local_feat_dim,
            ).cuda()

        # train the network
        scores, queue, local_queue = train(train_loader, model, optimizer, scaler, epoch, lr_schedule, queue, local_queue)
        training_stats.update(scores)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16:
                save_dict["scaler"] = scaler.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        if queue is not None:
            torch.save({"queue": queue}, queue_path)
        if local_queue is not None:
            torch.save({"local_queue": local_queue}, local_queue_path)


def train(train_loader, model, optimizer, scaler, epoch, lr_schedule, queue, local_queue):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    glb_losses = AverageMeter()
    local_losses = AverageMeter()
    l2g_losses = AverageMeter()

    model.train()
    use_the_queue = False

    end = time.time()
    for it, inputs in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            w = model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

            local_w = model.module.local_prototypes.weight.data.clone()
            local_w = nn.functional.normalize(local_w, dim=1, p=2)
            model.module.local_prototypes.weight.copy_(local_w)
            
            l2g_w = model.module.l2g_prototypes.weight.data.clone()
            l2g_w = nn.functional.normalize(l2g_w, dim=1, p=2)
            model.module.l2g_prototypes.weight.copy_(l2g_w)

        # ============ pytorch mixed precision ============
        with autocast(dtype=torch.float16, enabled=args.use_fp16):
        # ============ multi-res forward passes ... ============
            (embedding, output), (local_z, local_logit) = model(inputs)
            embedding = embedding.detach()
            local_z = local_z.detach()
            bs = inputs[0].size(0)

            # ============ swav loss ... ============
            global_loss = 0
            global_q = []
            for i, crop_id in enumerate(args.crops_for_assign):
                with torch.no_grad():
                    out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                    # time to use the queue
                    if queue is not None:
                        if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                            use_the_queue = True
                            out = torch.cat((torch.mm(
                                queue[i],
                                model.module.prototypes.weight.t()
                            ), out))
                        # fill the queue
                        queue[i, bs:] = queue[i, :-bs].clone()
                        queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                    # get assignments
                    q = distributed_sinkhorn(out)[-bs:]
                    global_q.append(q)

                # cluster assignment prediction
                subloss = 0
                for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                    x = output[bs * v: bs * (v + 1)] / args.temperature
                    subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                global_loss += subloss / (np.sum(args.nmb_crops) - 1)
            global_loss /= len(args.crops_for_assign)

            # ============ local loss ... ===============
            nmb_local_views = args.local_feat_dim // args.nmb_groups_heads
            local_bs = bs * nmb_local_views

            local_loss = 0
            for i, crop_id in enumerate(args.crops_for_assign):
                with torch.no_grad():
                    out = local_logit[local_bs * crop_id: local_bs * (crop_id + 1)].detach()

                    # time to use the local queue
                    if local_queue is not None:
                        if use_the_queue or not torch.all(local_queue[i, -1, :] == 0):
                            use_the_queue = True
                            out = torch.cat((torch.mm(
                                local_queue[i],
                                model.module.prototypes.weight.t()
                            ), out))
                        # fill the local queue
                        local_queue[i, local_bs:] = local_queue[i, :-local_bs].clone()
                        local_queue[i, :local_bs] = local_z[crop_id * local_bs: (crop_id + 1) * local_bs]

                        # get assignments
                    local_q = distributed_sinkhorn(out)[-local_bs:]
                    
                    # cluster assignment prediction
                subloss = 0
                for v in np.delete(np.arange(args.nmb_crops[0]), crop_id):
                    x = local_logit[local_bs * v: local_bs * (v + 1)] / args.temperature
                    x = shuffle_logits(x, nmb_local_views, bs)
                    subloss -= torch.mean(torch.sum(local_q * F.log_softmax(x, dim=1), dim=1))
                local_loss += subloss / (args.nmb_crops[0] - 1)
            local_loss /= len(args.crops_for_assign)
            
            # ============ Local 2 Global loss ... ============
            l2g_loss = 0
            for i, crop_id in enumerate(args.crops_for_assign):
                subloss = 0
                for v in np.delete(np.arange(args.nmb_crops[0]), crop_id):
                    x = local_z[local_bs * v: local_bs * (v + 1)]
                    x = concat_local_logits(x, nmb_local_views, bs)
                    l2g_logits = model.module.forward_l2g(x)
                    
                    for v_id in range(len(global_q)):
                        g_q = global_q[v_id]
                        p_l2g = l2g_logits / args.temperature
                        subloss -= torch.mean(torch.sum(g_q * F.log_softmax(p_l2g, dim=1), dim=1))
                l2g_loss += subloss / len(global_q)
            l2g_loss /= len(args.crops_for_assign)

            loss = (global_loss + local_loss + l2g_loss) / 3.0

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        scaler.step(optimizer)
        scaler.update()

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        glb_losses.update(global_loss.item(), inputs[0].shape[0])
        local_losses.update(local_loss.item(), inputs[0].shape[0] * (args.local_feat_dim // args.nmb_groups_heads))
        l2g_losses.update(l2g_loss.item(), inputs[0].shape[0])
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}\t"
                "GLB: {glb.val:.4f} ({glb.avg:.4f})\t"
                "ALG: {alg.val:.4f} ({alg.avg:.4f})\t"
                "L2G: {l2g.val:.4f} ({l2g.avg:.4f})\t".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.optim.param_groups[0]["lr"],
                    glb=glb_losses,
                    alg=local_losses,
                    l2g=l2g_losses,
                )
            )
    return (epoch, losses.avg), queue, local_queue


@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


if __name__ == "__main__":
    main()

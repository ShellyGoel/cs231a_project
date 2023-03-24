# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os
import random
import torch
import torch.backends.cudnn
import torch.utils.data
import pdb
import trimesh.voxel as tv
from trimesh.voxel.ops import matrix_to_marching_cubes
from pytorch3d.renderer import FoVPerspectiveCameras,FoVOrthographicCameras,Materials,MeshRenderer,MeshRasterizer,SoftPhongShader, TexturesVertex
from pytorch3d.structures import Meshes

import os
import sys
import torch
import numpy as np
import json

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex,
    TexturesAtlas,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer
)

import matplotlib.pyplot as plt
from skimage.io import imread
from utils import *

import plotly.graph_objs as go
import plotly.io as pio
import pytorch3d

# Util function for loading meshes
from pytorch3d.io import load_obj

from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.ops import sample_points_from_meshes

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex,
    TexturesAtlas,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer
)
from pytorch3d.ops.marching_cubes import marching_cubes
from pytorch3d.ops.cubify import cubify

from pytorch3d.io import save_obj
# -

import numpy as np
import torch
import pyrender
from scipy.spatial.transform import Rotation
import trimesh

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.test import test_net
from models.encoder import Encoder
from models.decoder import Decoder
from models.merger import Merger


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomFlip(),
        utils.data_transforms.RandomPermuteRGB(),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    num_workers=cfg.TRAIN.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
                                                  batch_size=1,
                                                  num_workers=1,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    merger = Merger(cfg)
    print('[DEBUG] %s Parameters in Encoder: %d.' % (dt.now(), utils.network_utils.count_parameters(encoder)))
    print('[DEBUG] %s Parameters in Decoder: %d.' % (dt.now(), utils.network_utils.count_parameters(decoder)))
    print('[DEBUG] %s Parameters in Merger: %d.' % (dt.now(), utils.network_utils.count_parameters(merger)))

    # Initialize weights of networks
    encoder.apply(utils.network_utils.init_weights)
    decoder.apply(utils.network_utils.init_weights)
    merger.apply(utils.network_utils.init_weights)

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(decoder.parameters(),
                                          lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        merger_solver = torch.optim.Adam(merger.parameters(), lr=cfg.TRAIN.MERGER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(decoder.parameters(),
                                         lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        merger_solver = torch.optim.SGD(merger.parameters(),
                                        lr=cfg.TRAIN.MERGER_LEARNING_RATE,
                                        momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
                                                                milestones=cfg.TRAIN.ENCODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver,
                                                                milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    merger_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(merger_solver,
                                                               milestones=cfg.TRAIN.MERGER_LR_MILESTONES,
                                                               gamma=cfg.TRAIN.GAMMA)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        merger = torch.nn.DataParallel(merger).cuda()

        
    # Set our device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        # Set up loss functions
        #pdb.set_trace()
    bce_loss = torch.nn.BCELoss()

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

        print('[INFO] %s Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
              (dt.now(), init_epoch, best_iou, best_epoch))

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    log_dir = output_dir % 'logs'
    ckpt_dir = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'test'))

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        encoder_losses = utils.network_utils.AverageMeter()

        # switch models to training mode
        encoder.train()
        decoder.train()
        merger.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_names, sample_names, rendering_images,
                        ground_truth_volumes, metadata) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)

            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volumes = utils.network_utils.var_or_cuda(ground_truth_volumes)

            encoder_input = {
                'rendering_images': rendering_images
            }
            #pdb.set_trace()
            #if cfg.GEOMETRIC.METADATA:
            encoder_input['metadata'] = metadata
                

            # Train the encoder, decoder, and merger
            image_features = encoder(encoder_input)
            raw_features, generated_volumes = decoder(image_features)

            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volumes = merger(raw_features, generated_volumes)
            else:
                generated_volumes = torch.mean(generated_volumes, dim=1)

            encoder_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10

            #calculate 2D reprojection loss
            azimuth = metadata[:, :, 0].squeeze()
            elevation = metadata[:, :, 1].squeeze()
            inplain_rot = metadata[:, :, 2].squeeze()
            distance = metadata[:, :, 3].squeeze()
            fov = metadata[:, :, 4].squeeze()


            ground_truth_imgs_list = []
            generated_imgs_list = []
            for i in range(len(azimuth)):
                azimuth_i = azimuth[i]
                elevation_i = elevation[i]
                inplain_rot_i = inplain_rot[i]
                distance_i = distance[i]
                fov_i = fov[i]

                #For ground truth volume -> image
                verts_gt, faces_gt = marching_cubes(ground_truth_volumes[i].unsqueeze(0), isolevel = 0.5)

                vert_gt = verts_gt[0]
                face_gt = faces_gt[0]
                
                
                if not torch.is_tensor(vert_gt):
                    
                    images_gt = torch.zeros((1, 224, 224, 4))
                    images_gen = torch.zeros((1, 224, 224, 4))
                    ground_truth_imgs_list.append(images_gt.cuda())
                    generated_imgs_list.append(images_gen.cuda())   
                    continue
                    
                verts_rgb_gt = torch.ones_like(vert_gt)[None]
                textures_gt = TexturesVertex(verts_features=verts_rgb_gt)
                mesh_gt = Meshes(verts=[vert_gt], faces=[face_gt], textures = textures_gt)

                R, T = look_at_view_transform(distance_i, elevation_i, azimuth_i, device='cuda')
                camera = FoVPerspectiveCameras(device='cuda', R=R, T=T, fov=fov_i)

                raster_settings = RasterizationSettings(image_size=224, blur_radius=0.0, faces_per_pixel=1)

                rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

                shader = SoftPhongShader(device='cuda', cameras=camera)

                renderer = MeshRenderer(rasterizer, shader)
                
                images_gt = renderer(mesh_gt, cameras=camera)

                #For generated volume -> image

                verts_gen, faces_gen = marching_cubes(generated_volumes[i].unsqueeze(0), isolevel = 0.5)

                vert_gen = verts_gen[0]
                face_gen = faces_gen[0]
                
                if not torch.is_tensor(vert_gen):
                    
                    images_gt = torch.zeros((1, 224, 224, 4))
                    images_gen = torch.zeros((1, 224, 224, 4))
                    ground_truth_imgs_list.append(images_gt.cuda())
                    generated_imgs_list.append(images_gen.cuda())
                    continue
                verts_rgb_gen = torch.ones_like(vert_gen)[None]
                textures_gen = TexturesVertex(verts_features=verts_rgb_gen)
                mesh_gen = Meshes(verts=[vert_gen], faces=[face_gen], textures = textures_gen)

                images_gen = renderer(mesh_gen, cameras=camera)

                ground_truth_imgs_list.append(images_gt)
                generated_imgs_list.append(images_gen)
        
            #calculate 2D reprojection loss between generated image and rendering image
            ground_truth_image = torch.stack(ground_truth_imgs_list, dim=0)
            generated_image = torch.stack(generated_imgs_list, dim=0)
            
            gt_image = torch.tensor(ground_truth_image, requires_grad=True)
            gen_image = torch.tensor(generated_image, requires_grad=True)
            
            #print('encoder loss: ', encoder_loss)
            reprojection_loss = torch.nn.MSELoss()(gen_image, gt_image)*10

            
            encoder_loss+=reprojection_loss
           
            # Gradient decent
            encoder.zero_grad()
            decoder.zero_grad()
            merger.zero_grad()
            
            encoder_loss.backward()          

            encoder_solver.step()
            decoder_solver.step()
            merger_solver.step()

            # Append loss to average metrics
            encoder_losses.update(encoder_loss.item())
            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            print('[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) EDLoss = %.4f' %
                  (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time.val,
                   data_time.val, encoder_loss.item()))

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx + 1)

        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        merger_lr_scheduler.step()

        # Tick / tock
        epoch_end_time = time()
        print('[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) EDLoss = %.4f' %
              (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, encoder_losses.avg))

        # Update Rendering Views
        if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
            n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
            train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
            print('[INFO] %s Epoch [%d/%d] Update #RenderingViews to %d' %
                  (dt.now(), epoch_idx + 2, cfg.TRAIN.NUM_EPOCHES, n_views_rendering))

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, decoder, merger)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth' % (epoch_idx + 1)),
                                                 epoch_idx + 1, encoder, encoder_solver, decoder, decoder_solver,
                                                 merger, merger_solver, best_iou, best_epoch)
        if iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            best_iou = iou
            best_epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(cfg, os.path.join(ckpt_dir, 'best-ckpt.pth'), epoch_idx + 1, encoder,
                                                 encoder_solver, decoder, decoder_solver, merger, merger_solver,
                                                 best_iou, best_epoch)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()






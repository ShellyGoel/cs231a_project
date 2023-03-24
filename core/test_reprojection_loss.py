# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

# +
import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data
import pdb
import os
import random
import torch
import torch.backends.cudnn
import torch.utils.data
import pdb
import trimesh.voxel as tv
from trimesh.voxel.ops import matrix_to_marching_cubes
import numpy as np
import torch
import pyrender
from scipy.spatial.transform import Rotation
import trimesh
import os
import sys
import torch
import numpy as np
import json

# # Data structures and functions for rendering
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

import numpy as np
import torch
import pyrender
from scipy.spatial.transform import Rotation
import trimesh
# -

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt

from models.encoder import Encoder
from models.decoder import Decoder
from models.merger import Merger


def test_net(cfg,
             epoch_idx=-1,
             output_dir=None,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None,
             merger=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                       num_workers=1,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        merger = Merger(cfg)

        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()
            merger = torch.nn.DataParallel(merger).cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    encoder_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    merger.eval()
    
    #pdb.set_trace()

    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume, metadata) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

            encoder_input = {
                'rendering_images': rendering_images
            }
            
            encoder_input['metadata'] = metadata

            # Test the encoder, decoder and merger
            image_features = encoder(encoder_input)
            raw_features, generated_volume = decoder(image_features)

            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volume = merger(raw_features, generated_volume)
            else:
                generated_volume = torch.mean(generated_volume, dim=1)

            encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10
                
            #calculate 2D reprojection loss
            azimuth = metadata[:, :, 0].squeeze()
            elevation = metadata[:, :, 1].squeeze()
            inplain_rot = metadata[:, :, 2].squeeze()
            distance = metadata[:, :, 3].squeeze()
            fov = metadata[:, :, 4].squeeze()


            ground_truth_imgs_list = []
            generated_imgs_list = []

            
            
            azimuth_i = azimuth
            elevation_i = elevation
            inplain_rot_i = inplain_rot
            distance_i = distance
            fov_i = fov

            #For ground truth volume -> image
            
            #pdb.set_trace()
            verts_gt, faces_gt = marching_cubes(ground_truth_volume, isolevel = 0.5)

            vert_gt = verts_gt[0]
            face_gt = faces_gt[0]
            
            if not torch.is_tensor(vert_gt):
                    
                images_gt = torch.zeros((224, 224, 4))
                images_gen = torch.zeros((224, 224, 4))
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

            verts_gen, faces_gen = marching_cubes(generated_volume, isolevel = 0.5)

            vert_gen = verts_gen[0]
            face_gen = faces_gen[0]
            
            #print(ground_truth_volume.shape)
            
            if not torch.is_tensor(vert_gen):
                    
                images_gt = torch.zeros((224, 224, 4))
                images_gen = torch.zeros((224, 224, 4))
                ground_truth_imgs_list.append(images_gt.cuda())
                generated_imgs_list.append(images_gen.cuda())
                
                continue

            verts_rgb_gen = torch.ones_like(vert_gen)[None]
            textures_gen = TexturesVertex(verts_features=verts_rgb_gen)
            mesh_gen = Meshes(verts=[vert_gen], faces=[face_gen], textures = textures_gen)

            images_gen = renderer(mesh_gen, cameras=camera)
    
            #pdb.set_trace()

            ground_truth_imgs_list.append(images_gt)
            generated_imgs_list.append(images_gen)
        
            #calculate 2D reprojection loss between generated image and rendering image
            ground_truth_image = torch.stack(ground_truth_imgs_list, dim=0)
            generated_image = torch.stack(generated_imgs_list, dim=0)
            
            gt_image = torch.tensor(ground_truth_image, requires_grad=True)
            gen_image = torch.tensor(generated_image, requires_grad=True)
            
            reprojection_loss = torch.nn.MSELoss()(gen_image, gt_image)*10

            encoder_loss+=reprojection_loss

            # Append loss and accuracy to average metrics
            encoder_losses.update(encoder_loss.item())

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            # Append generated volumes to TensorBoard
            if output_dir and sample_idx < 3:
                img_dir = output_dir % 'images'
                # Volume Visualization
                gv = generated_volume.cpu().numpy()
                rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(img_dir, 'test'),
                                                                              epoch_idx)
                test_writer.add_image('Test Sample#%02d/Volume Reconstructed' % sample_idx, rendering_views, epoch_idx, dataformats='HWC')
                gtv = ground_truth_volume.cpu().numpy()
                rendering_views = utils.binvox_visualization.get_volume_views(gtv, os.path.join(img_dir, 'test'),
                                                                              epoch_idx)
                test_writer.add_image('Test Sample#%02d/Volume GroundTruth' % sample_idx, rendering_views, epoch_idx, dataformats='HWC')

            # Print sample loss and IoU
            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f IoU = %s' %
                  (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, encoder_loss.item(),
                   ['%.4f' % si for si in sample_iou]))

    # Output testing results
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
        print(mean_iou)
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        if 'baseline' in taxonomies[taxonomy_id]:
            print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
        else:
            print('N/a', end='\t\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    print("mean_iou: ", mean_iou)
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('EncoderDecoder/IoU', max_iou, epoch_idx)

    return max_iou



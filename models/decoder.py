# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        self.use_metadata = cfg.GEOMETRIC.METADATA
        self.aux_channels = 0
        if (self.use_metadata):
            self.aux_channels += 1
        
        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256 + self.aux_channels, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )

        if self.use_metadata:
            self.metadata_layer = torch.nn.Sequential(
                torch.nn.Linear(3, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
                torch.nn.ReLU()
            )

    def forward(self, image_features, metadata=None):
        gen_voxels = []
        raw_features = []

        if (self.use_metadata):
            # print(image_features.size())  # torch.Size([batch_size, n_views, 128, 4, 4])
            image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
            image_features = torch.split(image_features, 1, dim=0)

            # print(metadata.size())  # torch.Size([batch_size, n_views, 3])
            metadata = metadata.permute(1, 0, 2).contiguous()
            metadata = torch.split(metadata, 1, dim=0)

            for features, md in zip(image_features, metadata):
                gen_voxel = features.view(-1, 256, 2, 2, 2)
                # print(gen_voxel.size())   # torch.Size([batch_size, 256, 2, 2, 2])

                metadata_features = self.metadata_layer(md.squeeze(dim=0))
                # print(metadata_features.size()) # torch.Size([batch_size, 1])
                metadata_features = metadata_features[:, None, None, :]
                # print(metadata_features.size()) # torch.Size([batch_size, 1, 1, 1])
                metadata_features = metadata_features.repeat(1, gen_voxel.size()[-3], gen_voxel.size()[-2], gen_voxel.size()[-1])
                metadata_features = metadata_features.unsqueeze(1)
                # print(metadata_features.size()) # torch.Size([batch_size, 1, 2, 2, 2])
                # print(gen_voxel.size()) # torch.Size([batch_size, 256, 2, 2, 2])

                gen_voxel = torch.cat((gen_voxel, metadata_features), dim=1)
                # print(gen_voxel.size()) # torch.Size([batch_size, 257, 2, 2, 2])

                gen_voxel = self.layer1(gen_voxel)
                # print(gen_voxel.size())   # torch.Size([batch_size, 128, 4, 4, 4])
                gen_voxel = self.layer2(gen_voxel)
                # print(gen_voxel.size())   # torch.Size([batch_size, 64, 8, 8, 8])
                gen_voxel = self.layer3(gen_voxel)
                # print(gen_voxel.size())   # torch.Size([batch_size, 32, 16, 16, 16])
                gen_voxel = self.layer4(gen_voxel)
                # print(gen_voxel.size())   # torch.Size([batch_size, 8, 32, 32, 32])
                raw_feature = gen_voxel
                gen_voxel = self.layer5(gen_voxel)
                # print(gen_voxel.size())   # torch.Size([batch_size, 1, 32, 32, 32])
                raw_feature = torch.cat((raw_feature, gen_voxel), dim=1)
                # print(raw_feature.size()) # torch.Size([batch_size, 9, 32, 32, 32])
                gen_voxels.append(torch.squeeze(gen_voxel, dim=1))
                raw_features.append(raw_feature)        
        else:
            image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
            image_features = torch.split(image_features, 1, dim=0)
            for features in image_features:
                gen_voxel = features.view(-1, 256, 2, 2, 2)
                # print(gen_voxel.size())   # torch.Size([batch_size, 256, 2, 2, 2])
                gen_voxel = self.layer1(gen_voxel)
                # print(gen_voxel.size())   # torch.Size([batch_size, 128, 4, 4, 4])
                gen_voxel = self.layer2(gen_voxel)
                # print(gen_voxel.size())   # torch.Size([batch_size, 64, 8, 8, 8])
                gen_voxel = self.layer3(gen_voxel)
                # print(gen_voxel.size())   # torch.Size([batch_size, 32, 16, 16, 16])
                gen_voxel = self.layer4(gen_voxel)
                # print(gen_voxel.size())   # torch.Size([batch_size, 8, 32, 32, 32])
                raw_feature = gen_voxel
                gen_voxel = self.layer5(gen_voxel)
                # print(gen_voxel.size())   # torch.Size([batch_size, 1, 32, 32, 32])
                raw_feature = torch.cat((raw_feature, gen_voxel), dim=1)
                # print(raw_feature.size()) # torch.Size([batch_size, 9, 32, 32, 32])
                gen_voxels.append(torch.squeeze(gen_voxel, dim=1))
                raw_features.append(raw_feature)

        gen_voxels = torch.stack(gen_voxels).permute(1, 0, 2, 3, 4).contiguous()
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        # print(gen_voxels.size())        # torch.Size([batch_size, n_views, 32, 32, 32])
        # print(raw_features.size())      # torch.Size([batch_size, n_views, 9, 32, 32, 32])
        return raw_features, gen_voxels
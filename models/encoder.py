# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        aux_channels = 0
        if (cfg.GEOMETRIC.METADATA):
            aux_channels += 1
        if (cfg.GEOMETRIC.INTRINSIC):
            aux_channels += 1
        if (cfg.GEOMETRIC.EXTRINSIC):
            aux_channels += 1
        if (cfg.GEOMETRIC.DEPTH):
            aux_channels += 1

        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512 + aux_channels, 512, kernel_size=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=4)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU()
        )
        self.metadata_layer = torch.nn.Sequential(
            torch.nn.Linear(5, 64),
            torch.nn.Linear(64, 1),
            torch.nn.ELU()
        )

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False

    def forward(self, encoder_input):
        rendering_images = encoder_input['rendering_images']
        metadata = encoder_input['metadata']
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        # print(metadata.size())  # torch.Size([batch_size, n_views, 5])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        metadata = metadata.permute(1, 0, 2).contiguous()
        metadata = torch.split(metadata, 1, dim=0)

        image_features = []

        for img, md in zip(rendering_images, metadata):
            metadata_features = self.metadata_layer(md.squeeze(dim=0))
            # print(metadata_features.size()) # torch.Size([batch_size, 1])
            features = self.vgg(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])

            metadata_features = metadata_features.unsqueeze(1).repeat(1, features.size()[-2], features.size()[-1])
            metadata_features = metadata_features.unsqueeze(1)
            # print(metadata_features.size()) # torch.Size([batch_size, 1, 28, 28])

            features = torch.cat((features, metadata_features), dim=1)
            # print(features.size()) # torch.Size([batch_size, 513, 28, 28])

            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 256, 6, 6])
            features = self.layer3(features)
            # print(features.size())    # torch.Size([batch_size, 128, 4, 4])
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 128, 4, 4])
        return image_features


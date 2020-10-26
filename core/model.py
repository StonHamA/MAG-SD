import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

EPSILON = 1e-12
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)


        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix



class res50Encoder(nn.Module):
    def __init__(self, config):
        super(res50Encoder, self).__init__()

        # load backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        self.fc = nn.Linear(2048*config.attention_map_num, config.class_num, bias=False)
        # self.fc_bone = nn.Linear(2048, config.class_num, bias=True)

        # features
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_encoder = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                         resnet.layer1, resnet.layer2)
        self.resnet_encoder3 = resnet.layer3
        self.resnet_encoder4 = resnet.layer4

        self.bap = BAP(pool='GAP')

        self.attention_module = DblAttentionModule(config)
        # self.attention_module = SimpleAttentionModule(config)
        # self.attention_module = TriAttentionModule(config)

        self.M = config.attention_map_num
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x, training=True):
        batch_size = x.size(0)
        # 2&1atten
        features_1 = self.resnet_encoder3(self.resnet_encoder(x))
        features_2 = self.resnet_encoder4(features_1)
        attention_maps = self.attention_module(features_1, features_2) #2atten
        # attention_maps = self.attention_module(features_2) #1atten
        feature_matrix = self.bap(features_2, attention_maps)

        # 3atten
        # features_3 = self.resnet_encoder(x)
        # features_2 = self.resnet_encoder3(features_3)
        # features_1 = self.resnet_encoder4(features_2)
        # attention_maps = self.attention_module(features_3,features_2,features_1)
        # feature_matrix = self.bap(features_1, attention_maps)

        logits = self.fc(feature_matrix*100)
        # GAP/GMP experiments
        # logits_bone = self.fc_bone(torch.squeeze(self.GMP(features_2)))
        # attention map 4 augment
        if training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                # print(attention_weights)
                k_index = np.random.choice(self.M, 3, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 3, H, W) -3 types of augs
        else:
            # Object Localization Am = mean(Ak)
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)


        return logits, features_1, attention_map

class SimpleAttentionModule(nn.Module):
    def __init__(self, config):
        super(SimpleAttentionModule, self).__init__()
        # attention
        self.attention_layer = nn.Sequential(nn.Conv2d(2048, config.attention_map_num, kernel_size=1),
                                             nn.BatchNorm2d(config.attention_map_num, eps=0.001),
                                             nn.ReLU(inplace=True))

    def forward(self, x):
        return self.attention_layer(x)

class TriAttentionModule(nn.Module):

    def __init__(self, config):
        super(TriAttentionModule, self).__init__()
        self.pixel_shuffel_upsample = nn.PixelShuffle(2)
        self.pixel_shuffel_upsample2 = nn.PixelShuffle(2)
        self.ReLU = nn.ReLU(inplace=True)
        self.attention_texture_high = nn.Sequential(nn.Conv2d(512, config.attention_map_num, kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(config.attention_map_num, eps=0.001),
                                             nn.ReLU(inplace=True))
        self.attention_texture = nn.Sequential(nn.Conv2d(1024, config.attention_map_num, kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(config.attention_map_num, eps=0.001),
                                             nn.ReLU(inplace=True))
        self.attention_target = nn.Sequential(nn.Conv2d(2048, config.attention_map_num, kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(config.attention_map_num, eps=0.001),
                                             nn.ReLU(inplace=True))
        self.avgpool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=4,stride=4)
    def forward(self, x3, x2, x1):
        # print(x.size())
        target_map = self.attention_target(x1)  # 32 channels, size
        # up2 = self.pixel_shuffel_upsample(x1) # 512 chs, size*2
        texture_map = self.attention_texture(x2)
        texture_map_high = self.attention_texture_high(x3)
        # attention_output = texture_map + F.interpolate(target_map, scale_factor=2, mode='bilinear')
        attention_output = target_map + self.avgpool(texture_map)+ self.avgpool2(texture_map_high)
        return attention_output

class DblAttentionModule(nn.Module):

    def __init__(self, config):
        super(DblAttentionModule, self).__init__()
        self.pixel_shuffel_upsample = nn.PixelShuffle(2)
        self.pixel_shuffel_upsample2 = nn.PixelShuffle(2)
        self.ReLU = nn.ReLU(inplace=True)
        self.attention_texture = nn.Sequential(nn.Conv2d(1024, config.attention_map_num, kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(config.attention_map_num, eps=0.001),
                                             nn.ReLU(inplace=True))
        self.attention_target = nn.Sequential(nn.Conv2d(2048, config.attention_map_num, kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(config.attention_map_num, eps=0.001),
                                             nn.ReLU(inplace=True))
        self.avgpool = nn.AvgPool2d(kernel_size=2,stride=2)
    def forward(self, x2, x1):
        # print(x.size())
        target_map = self.attention_target(x1)  # 32 channels, size
        # up2 = self.pixel_shuffel_upsample(x1) # 512 chs, size*2
        texture_map = self.attention_texture(x2)
        # attention_output = texture_map + F.interpolate(target_map, scale_factor=2, mode='bilinear')
        attention_output = target_map + self.avgpool(texture_map)
        return attention_output



class res50(nn.Module):
    def __init__(self, config):
        super(res50, self).__init__()
        # # load backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        self.fc = nn.Linear(2048, config.class_num, bias=False)

        # # resnet feature
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_encoder = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                         resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.GAP = nn.AdaptiveAvgPool2d(1)
    def forward(self,x):
        map_out = self.resnet_encoder(x)
        x = self.GAP(map_out)
        x = self.fc(x.squeeze())
        return x, map_out

class res18(nn.Module):
    def __init__(self, config):
        super(res18, self).__init__()
        # # load backbone and optimize its architecture
        resnet = torchvision.models.resnet18(pretrained=True)
        self.fc = nn.Linear(512, config.class_num, bias=False)

        # # resnet feature
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_encoder = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                         resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.GAP = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        x = self.resnet_encoder(x)
        x = self.GAP(x)
        x = self.fc(x.squeeze())

        return x, x



class vgg16(nn.Module):
    def __init__(self, config):
        super(vgg16, self).__init__()
        # VGG16 feature
        model = torchvision.models.vgg16(pretrained=True)
        model.features[0] =  nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[-1] = nn.Linear(in_features=4096, out_features= config.class_num, bias=True)
        self.feature_generator = model.features
        self.classifier = nn.Sequential(model.classifier)
        self.model = model

    def forward(self,x):
        map_out = self.feature_generator(x)
        x = self.model(x)
        return x, map_out

class I3(nn.Module):
    def __init__(self, config):
        super(I3, self).__init__()
        # inception_v3 feature
        model = torchvision.models.inception_v3(pretrained=True, init_weights=True)
        model.fc = torch.nn.Linear(in_features=2048, out_features=config.class_num, bias=True)
        model.aux_logits = False
        self.model = model

    def forward(self,x):
        x = self.model(x)
        return x, x

class densenet121(nn.Module):
    def __init__(self, config):
        super(densenet121, self).__init__()
        # inception_v3 feature
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = torch.nn.Linear(in_features=1024, out_features=config.class_num, bias=True)
        self.model = model

    def forward(self,x):
        x = self.model(x)
        return x, x
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FeatExt_ref(nn.Module):
    def __init__(self):
        super(FeatExt_ref, self).__init__()
        base_channels = 8
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU())

        self.conv0_0 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, padding=1)
        self.conv0_1 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, relu=False, padding=1)
        self.conv0_2 = Conv2d(base_channels * 1, base_channels * 1, 1, stride=1, relu=False)

        self.conv1_0 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, padding=1)
        self.conv1_1 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, relu=False, padding=1)

        self.conv2_0 = Conv2d(base_channels * 1, base_channels * 2, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(base_channels * 1, base_channels * 2, 1, stride=2, relu=False)   # half

        self.conv3_0 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)    # half
        self.conv4_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)
        self.conv4_2 = Conv2d(base_channels * 2, base_channels * 4, 1, stride=2, relu=False)

        self.conv5_0 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv5_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv6_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1, bias=False)  # two times
        self.conv6_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)
        self.conv6_2 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv6_3 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.conv7_0 = nn.ConvTranspose2d(base_channels * 2, base_channels * 1, 3, 2, 1, 1, bias=False)  # two times
        self.conv7_1 = nn.Conv2d(base_channels * 2, base_channels * 1, 3, 1, 1, bias=False)
        self.conv7_2 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, padding=1)
        self.conv7_3 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, relu=False, padding=1)


        self.final_conv_1 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.final_conv_2 = nn.Conv2d(16, 16, 3, 1, 1, bias=False)
        self.final_conv_3 = nn.Conv2d(8,8,3, 1, 1, bias=False)

    def forward(self, x):
        x = self.init_conv(x)

        residual = x
        x = self.conv0_1(self.conv0_0(x))
        x += self.conv0_2(residual)
        x = nn.ReLU(inplace=True)(x)

        residual = x
        x = self.conv1_1(self.conv1_0(x))
        x += residual
        out3 = nn.ReLU(inplace=True)(x)


        residual = out3
        x = self.conv2_1(self.conv2_0(out3))
        x += self.conv2_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv3_1(self.conv3_0(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        residual = out2
        x = self.conv4_1(self.conv4_0(out2))
        x += self.conv4_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv5_1(self.conv5_0(x))
        x += residual
        out1 = nn.ReLU(inplace=True)(x)

        x = self.conv6_0(out1)
        x = torch.cat([x, out2], 1)
        x = self.conv6_1(x)
        residual = x
        x = self.conv6_3(self.conv6_2(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        x = self.conv7_0(out2)
        x = torch.cat([x, out3], 1)
        x = self.conv7_1(x)
        residual = x
        x = self.conv7_3(self.conv7_2(x))
        x += residual
        out3 = nn.ReLU(inplace=True)(x)


        outputs = {}
        outputs["stage1"] = self.final_conv_1(out1)
        outputs["stage2"] = self.final_conv_2(out2)
        outputs["stage3"] = self.final_conv_3(out3)
        return outputs

class Refinement(nn.Module):
    def __init__(self, feat_channels):
        super(Refinement, self).__init__()
        base_channels = 8

        self.conv1_0 = nn.Sequential(
            Conv2d(1, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1))

        self.conv1_2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 3, 2, 1, 1, bias=False) # half

        self.conv2_0 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 1, stride=2, relu=False)

        self.conv3_0 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1, bias=False)  # Double
        self.conv4_1 = nn.Conv2d(feat_channels+base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)
        self.conv4_2 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv4_3 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.final_conv = nn.Conv2d(base_channels * 2, 1, 3, padding=1, bias=False)

    def forward(self, depth, img_feat):
        # standardize
        depth_mean = torch.mean(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_std = torch.std(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth = (depth.unsqueeze(1) - depth_mean.unsqueeze(-1).unsqueeze(-1)) / depth_std.unsqueeze(-1).unsqueeze(-1)
        depth_min, _ = torch.min(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_max,_ = torch.max(depth.reshape(depth.shape[0],-1), -1, keepdim=True)

        # depth size half.
        depth_feat = self.conv1_2(self.conv1_0(depth))

        # concat
        cat = torch.cat((img_feat, depth_feat), dim=1)


        residual = cat
        x = self.conv2_1(self.conv2_0(cat))
        x += self.conv2_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv3_1(self.conv3_0(x))
        x += residual
        out1 = nn.ReLU(inplace=True)(x)

        x = self.conv4_0(out1)
        x = torch.cat([x, cat], 1)
        x = self.conv4_1(x)
        residual = x
        x = self.conv4_3(self.conv4_2(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        res = self.final_conv(out2)

        res_ = torch.zeros_like(res)
        for i in range(res.shape[0]):
            res_[i] = torch.clamp(res[i], min=depth_min[i].cpu().item(), max=depth_max[i].cpu().item())
        depth = (res_ + F.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=False)) * depth_std.unsqueeze(-1).unsqueeze(-1) + depth_mean.unsqueeze(-1).unsqueeze(-1)

        return res_.squeeze(1), depth.squeeze(1)

class Refinement_Res(nn.Module):
    def __init__(self, feat_channels):
        super(Refinement_Res, self).__init__()
        base_channels = 8

        self.conv1_0 = nn.Sequential(
            Conv2d(1, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1))


        self.conv2_0 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 1, stride=2, relu=False)

        self.conv3_0 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1, bias=False)  # Double
        self.conv4_1 = nn.Conv2d(feat_channels+base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)
        self.conv4_2 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv4_3 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.final_conv = nn.Conv2d(base_channels * 2, 1, 3, padding=1, bias=False)

    def forward(self, depth, img_feat):
        # if the input size of depth is w,h; and the size of img_feat is 2w, 2h.
        # standardize
        depth_mean = torch.mean(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_std = torch.std(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth = (depth.unsqueeze(1) - depth_mean.unsqueeze(-1).unsqueeze(-1)) / depth_std.unsqueeze(-1).unsqueeze(-1)
        depth_min, _ = torch.min(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_max,_ = torch.max(depth.reshape(depth.shape[0],-1), -1, keepdim=True)

        # depth size double, depth_feature is 2w, 2h
        depth_feat = self.conv1_0(depth)

        # concat, channel concat.
        cat = torch.cat((img_feat, depth_feat), dim=1)

        # A residual block, the feature is the half of the input
        residual = cat
        x = self.conv2_1(self.conv2_0(cat))  # half, w, h
        x += self.conv2_2(residual)  # half, w, h
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv3_1(self.conv3_0(x))  # size not change
        x += residual
        out1 = nn.ReLU(inplace=True)(x)

        x = self.conv4_0(out1)  #  double, 2w, 2h.
        x = torch.cat([x, cat], 1)
        x = self.conv4_1(x)  # not change, 2w, 2h.
        residual = x
        x = self.conv4_3(self.conv4_2(x))  # not change, 2w, 2h
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        res = self.final_conv(out2)  # not change, the channel is set to 1.

        res_ = torch.zeros_like(res)
        for i in range(res.shape[0]):
            res_[i] = torch.clamp(res[i], min=depth_min[i].cpu().item(), max=depth_max[i].cpu().item())
        depth = (res_ + depth * depth_std.unsqueeze(-1).unsqueeze(-1) + depth_mean.unsqueeze(-1).unsqueeze(-1))
        return res_.squeeze(1), depth.squeeze(1)

class OffsetNet(nn.Module):
    def __init__(self, residual):
        super(OffsetNet, self).__init__()
        if residual != 0:
           self.residual_net = Refinement_Res(residual)

    def forward(self, inputs, feat):

        depth_init_res = inputs["depth"]
        inputs["depth_init"] = inputs["depth"]
        res, depth = self.residual_net(depth_init_res.detach(), feat)
        inputs["depth"] = depth
        inputs["res"] = res

        return inputs
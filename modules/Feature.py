import torch
import torch.nn as nn
import torch.nn.functional as F
class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        assert init_method in ["kaiming", "xavier"]
        self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return
def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        # 使用全连接层来生成通道的注意力系数
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)  # 第一个全连接层，用于降维
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)  # 第二个全连接层，用于恢复维度
        self.sigmoid = nn.Sigmoid()  # 激活函数，用于生成最终的注意力系数

    def forward(self, x):
        # Squeeze操作：按通道维度进行全局平均池化，得到每个通道的全局特征
        b, c, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1).view(b, c)  # 将每个通道的特征压缩为一个值

        # Excitation操作：通过两个全连接层生成每个通道的注意力系数
        excitation = self.fc1(squeeze)
        excitation = F.relu(excitation)  # 使用ReLU激活
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(b, c, 1, 1)  # 使用Sigmoid激活并恢复成图像形状

        # 进行通道加权
        return x * excitation

class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = SEBlock(in_channels, reduction_ratio)
        self.spatial_attention = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # Channel Attention
        x = self.channel_attention(x)

        # Spatial Attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        spatial_attention_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attention_map = self.spatial_attention(spatial_attention_input)
        return x * spatial_attention_map


class DynamicSobelKernel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DynamicSobelKernel, self).__init__()
        # 初始化传统的 Sobel 核作为可微的参数
        sobel_kernel_0 = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_45 = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_90 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_135 = torch.tensor([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

        # 将 Sobel 核作为初始值，并且使用 nn.Parameter 使其可学习
        self.kernel_x = nn.Parameter(sobel_kernel_0.expand(in_channels, 1, 3, 3))
        self.kernel_45 = nn.Parameter(sobel_kernel_45.expand(in_channels, 1, 3, 3))
        self.kernel_y = nn.Parameter(sobel_kernel_90.expand(in_channels, 1, 3, 3))
        self.kernel_135 = nn.Parameter(sobel_kernel_135.expand(in_channels, 1, 3, 3))

        # 可学习的卷积层用于学习注意力权重
        self.attention_layer = Conv2d(in_channels * 4, 4, kernel_size=1, stride=1, padding=0)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 提取四方向梯度
        grad_x = F.conv2d(x, self.kernel_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, self.kernel_y, padding=1, groups=x.size(1))
        grad_45 = F.conv2d(x, self.kernel_45, padding=1, groups=x.size(1))
        grad_135 = F.conv2d(x, self.kernel_135, padding=1, groups=x.size(1))

        # 将四个方向的梯度特征堆叠在一起
        gradients = torch.cat([grad_x, grad_y, grad_45, grad_135], dim=1)  # (B, 4, H, W)

        # 使用注意力机制计算每个方向的注意力权重
        attention_weights = self.attention_layer(gradients)  # (B, 4, H, W)
        attention_weights = F.softmax(attention_weights, dim=1)  # 使用 Softmax 归一化

        # 对四个方向的梯度应用注意力权重
        weighted_gradients = torch.sum(attention_weights * gradients, dim=1, keepdim=True)
        gradients = self.conv1(weighted_gradients)

        return gradients


class MultiScaleFrequencyBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFrequencyBranch, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 多尺度高频卷积
        self.conv_high_small = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_high_large = Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

        # 单尺度低频卷积
        self.conv_low = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)



    def forward(self, x):
        # 计算傅里叶变换
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft)

        # 高低频分离
        magnitude = torch.abs(fft_shift)
        high_freq = torch.where(magnitude > magnitude.mean(), magnitude, torch.tensor(0.0, device=x.device))
        low_freq = magnitude - high_freq

        # 多尺度高频卷积
        high_features_small = self.conv_high_small(high_freq.unsqueeze(1))
        high_features_large = self.conv_high_large(high_freq.unsqueeze(1))
        # 融合多尺度高频特征
        high_features = high_features_small + high_features_large

        # 单尺度低频卷积
        low_features = self.conv_low(low_freq.unsqueeze(1))

        # 融合高低频特征
        fused_features = high_features + low_features

        return fused_features


class AttentionInteraction(nn.Module):
    def __init__(self, num_branches, out_channels):
        super(AttentionInteraction, self).__init__()
        self.attention = nn.Conv2d(num_branches * out_channels, num_branches, kernel_size=1)

    def forward(self, branches):
        # 拼接所有分支特征
        concatenated = torch.cat(branches, dim=1)
        # 计算每个分支的权重
        weights = torch.softmax(self.attention(concatenated), dim=1)  # 权重归一化
        # 为每个分支加权
        weighted_features = [branches[i] * weights[:, i:i+1, :, :] for i in range(len(branches))]
        # 融合特征
        fused_features = sum(weighted_features)
        return fused_features


class AdaptiveFrequencyBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveFrequencyBranch, self).__init__()
        # 多尺度卷积
        self.conv_high_small = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_high_large = Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv_low = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.final_cov = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 计算傅里叶变换
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft)

        # 动态高低频分离
        magnitude = torch.abs(fft_shift)
        threshold = magnitude.mean() + magnitude.std()  # 动态阈值
        high_freq = torch.where(magnitude > threshold, magnitude, torch.tensor(0.0, device=x.device))
        low_freq = magnitude - high_freq

        # 多尺度高频处理
        high_features_small = self.conv_high_small(high_freq)
        high_features_large = self.conv_high_large(high_freq)
        high_features = high_features_small + high_features_large

        # 低频处理
        low_features = self.conv_low(low_freq)

        # 融合
        fused_features = high_features + low_features
        fused_features = self.final_cov (fused_features)

        return fused_features


class MultiBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiBranch, self).__init__()
        # 常规分支
        self.normal_branch = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )

        # 动态边缘分支
        self.edge_branch = DynamicSobelKernel(in_channels, out_channels)

        # 自适应频率分支
        self.frequency_branch = AdaptiveFrequencyBranch(in_channels, out_channels)

        # 分支融合权重动态化
        self.branch_weights = nn.Parameter(torch.ones(3))  # 初始化权重为 1

        # 特征融合与注意力机制
        self.interaction = AttentionInteraction(num_branches=3, out_channels=out_channels)
        self.cbam_block = CBAMBlock(out_channels)

    def forward(self, x):
        # 提取三分支特征
        normal_features = self.normal_branch(x)
        edge_features = self.edge_branch(x)
        frequency_features = self.frequency_branch(x)

        # 分支权重调整（通过 softmax 归一化）
        weights = torch.softmax(self.branch_weights, dim=0)
        normal_features = normal_features * weights[0]
        edge_features = edge_features * weights[1]
        frequency_features = frequency_features * weights[2]

        # 交互融合
        fused_features = self.interaction([normal_features, edge_features, frequency_features])

        # # 注意力优化
        # out = self.cbam_block(fused_features)
        return fused_features


class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class FeatureNet3(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet+sobel"):
        super(FeatureNet3, self).__init__()
        assert arch_mode in ["unet+sobel"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            MultiBranch(1, base_channels),
            # Conv2d(1, base_channels, 3, 1, padding=1),

            Conv2d(base_channels , base_channels , 3, 1, padding=1),
            # HybridConvModule(3, base_channels, downsample=False),
        )
        self.conv0_res = Conv2d(1, base_channels,3,1,padding=1)

        self.conv1 = nn.Sequential(
            # FusionModule(base_channels, base_channels * 2, downsample=True),
            # Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            # HybridConvModule(base_channels, base_channels * 2, downsample=True),
            # HybridConvModule(base_channels * 2, base_channels * 2),
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )
        self.conv1_res = Conv2d(base_channels, base_channels*2, 3,2,padding=1)

        self.conv2 = nn.Sequential(
            # FusionModule(base_channels * 2, base_channels * 4, downsample=True),
            # Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            # HybridConvModule(base_channels * 2, base_channels * 4, downsample=True),
            # HybridConvModule(base_channels * 4, base_channels * 4),
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        self.conv2_res = Conv2d(base_channels*2, base_channels*4, 3, 2,padding=1)

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]


        self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
        self.deconv1_res = Deconv2d(base_channels *4, base_channels*2, 3,2)

        self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)
        self.deconv2_res = Deconv2d(base_channels*2, base_channels,3,2)


        self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
        self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
        self.out_channels.append(2 * base_channels)
        self.out_channels.append(base_channels)

    def forward(self, x):
        # stage1 is the smallest size , and stage3 is the largest.
        residual = x
        conv0 = self.conv0(x)
        conv0 += self.conv0_res(residual)


        residual = conv0
        conv1 = self.conv1(conv0)
        conv1 += self.conv1_res(residual)


        residual = conv1
        conv2 = self.conv2(conv1)
        conv2 += self.conv2_res(residual)

        # print('conv2',conv2.shape)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        # print('out', out.shape)
        outputs["stage1"] = out

        residual = conv2
        intra_feat = self.deconv1(conv1, intra_feat)
        intra_feat += self.deconv1_res(residual)

        out = self.out2(intra_feat)
        outputs["stage2"] = out

        residual = intra_feat
        intra_feat = self.deconv2(conv0, intra_feat)
        intra_feat += self.deconv2_res(residual)

        out = self.out3(intra_feat)
        outputs["stage3"] = out

        return outputs
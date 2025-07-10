import torch
from modules.warping import RPC_Photo2Obj, RPC_Obj2Photo
import torch.nn.functional as F

def _bilinear_sample(im, x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)

    batch_size, height, width, channels = im.shape

    x, y = x.float(), y.float()
    max_y = int(height - 1)
    max_x = int(width - 1)

    x = (x + 1.0) * (width - 1.0) / 2.0
    y = (y + 1.0) * (height - 1.0) / 2.0

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    mask = (x0 >= 0) & (x1 <= max_x) & (y0 >= 0) & (y1 <= max_y)
    mask = mask.float()

    x0 = torch.clamp(x0, 0, max_x)
    x1 = torch.clamp(x1, 0, max_x)
    y0 = torch.clamp(y0, 0, max_y)
    y1 = torch.clamp(y1, 0, max_y)
    dim2 = width
    dim1 = width * height

    base = torch.arange(batch_size, device=im.device) * dim1
    base = base.reshape(-1, 1).repeat(1, height * width).reshape(-1)

    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    im_flat = im.reshape(-1, channels).float()
    pixel_a = im_flat[idx_a]
    pixel_b = im_flat[idx_b]
    pixel_c = im_flat[idx_c]
    pixel_d = im_flat[idx_d]

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (1.0 - (y1.float() - y))
    wc = (1.0 - (x1.float() - x)) * (y1.float() - y)
    wd = (1.0 - (x1.float() - x)) * (1.0 - (y1.float() - y))
    wa, wb, wc, wd = wa.unsqueeze(1), wb.unsqueeze(1), wc.unsqueeze(1), wd.unsqueeze(1)

    output = wa * pixel_a + wb * pixel_b + wc * pixel_c + wd * pixel_d
    output = output.reshape(batch_size, height, width, channels)
    mask = mask.reshape(batch_size, height, width, 1)
    return output, mask


def _spatial_transformer(img, coords):
    """
    A wrapper over _bilinear_sample(), taking absolute coords as input.
    img: [B, H, W, C]
    coords: [B, H, W, 2]，表示在源图像中要采样的位置
    """
    img_height = img.shape[1]
    img_width = img.shape[2]

    px = coords[:, :, :, :1]  # x 坐标
    py = coords[:, :, :, 1:]  # y 坐标

    # 归一化为 [-1, 1] 区间，grid_sample 要求的坐标格式
    px = px / (img_width - 1) * 2.0 - 1.0
    py = py / (img_height - 1) * 2.0 - 1.0

    # 送入 bilinear 采样器
    output_img, mask = _bilinear_sample(img, px, py)
    return output_img, mask

# 将source 上的影像 通过RPC 重投影到 reference 上，之后可以对比两者差异，自监督。
def inverse_warping_rpc(img, src_rpc, ref_rpc, depth, coef):
    # img: [B, H, W, C]（右影像）
    # depth: [B, H, W]，表示从参考视角推理出的每像素高程
    # src_rpc/ref_rpc: [B, 170]，RPC系数
    # coef: [B, H*W, 20] 的缓存张量，避免重复内存分配

    B, H, W, C = img.shape
    device = img.device

    # 生成像素坐标网格
    y, x = torch.meshgrid(
        torch.arange(0, H, dtype=torch.double, device=device),
        torch.arange(0, W, dtype=torch.double, device=device),
        indexing='ij'
    )
    x = x.contiguous().view(1, -1).repeat(B, 1)
    y = y.contiguous().view(1, -1).repeat(B, 1)
    h = depth.view(B, -1).double()     # [B, H*W]

    # Step 1: 从 ref 图像坐标系投影到地理坐标
    lat, lon = RPC_Photo2Obj(x, y, h, ref_rpc, coef)  # [B, H*W]

    # Step 2: 从地理坐标重投影到 src 图像坐标系
    samp, line = RPC_Obj2Photo(lat, lon, h, src_rpc, coef)  # [B, H*W]

    # Step 3: 归一化采样坐标 [-1, 1]
    samp = samp.float()
    line = line.float()
    proj_x_normalized = samp / ((W - 1) / 2) - 1
    proj_y_normalized = line / ((H - 1) / 2) - 1

    grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=2)  # [B, H*W, 2]
    grid = grid.view(B, H, W, 2)

    # Step 4: 重采样图像
    warped_right, mask = _spatial_transformer(img, grid)

    return warped_right, mask

def rpc_warping(src_fea, src_rpc, ref_rpc, depth_values):
    # src_fea: [B, C, H, W]
    # src_rpc: [B, 170]
    # ref_rpc: [B, 170]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]

    # import time
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    # with torch.no_grad():

    y, x = torch.meshgrid(
        torch.arange(0, height, dtype=torch.double, device=src_fea.device),
        torch.arange(0, width, dtype=torch.double, device=src_fea.device),
        indexing='ij'
    )

    y, x = y.contiguous(), x.contiguous()
    y = y.view(1, 1, height, width).repeat(batch, num_depth, 1, 1) # (B, ndepth, H, W)
    x = x.view(1, 1, height, width).repeat(batch, num_depth, 1, 1)

    if len(depth_values.shape) == 2:
        h = depth_values.view(batch, num_depth, 1, 1).double().repeat(1, 1, height, width) # (B, ndepth, H, W)
    else:
        h = depth_values # (B, ndepth, H, W)

    x = x.view(batch, -1)
    y = y.view(batch, -1)
    h = h.view(batch, -1)
    h = h.double()

    # start = time.time()
    lat, lon = RPC_Photo2Obj(x, y, h, ref_rpc)
    samp, line = RPC_Obj2Photo(lat, lon, h, src_rpc) # (B, ndepth*H*W)
    # end = time.time()

    # print(torch.mean(samp - x), torch.var(samp - x))
    # print(torch.mean(line - y), torch.var(line - y))

    samp = samp.float()
    line = line.float()

    # == 构造 mask ==
    # 只保留那些投影坐标在 src_fea 范围内的点
    valid_mask = (samp >= 0) & (samp <= width - 1) & (line >= 0) & (line <= height - 1)
    valid_mask = valid_mask.float()  # [B, H*W] or [B, N*H*W]

    proj_x_normalized = samp / ((width - 1) / 2) - 1
    proj_y_normalized = line / ((height - 1) / 2) - 1
    proj_x_normalized = proj_x_normalized.view(batch, num_depth, height * width)
    proj_y_normalized = proj_y_normalized.view(batch, num_depth, height * width)

    proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)
    grid = proj_xy

    # === bilinear sample ===
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2),
                                   mode='bilinear', padding_mode='zeros', align_corners=True)

    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    warped_src_fea = warped_src_fea.squeeze(2)

    # === reshape mask ===
    mask = valid_mask.view(batch, num_depth, height, width)
    mask = mask.squeeze(1)  # 只保留 2D（如果你只有 1 层深度）

    return warped_src_fea, mask
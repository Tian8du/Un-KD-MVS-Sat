
import torch
import torch.nn.functional as F


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]

    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    # depth_values = -depth_values ???
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj)) # Tcw
        #proj = torch.matmul(torch.inverse(src_proj), ref_proj)   # Twc
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        # y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
        #                        torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = torch.meshgrid(
            torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
            torch.arange(0, width, dtype=torch.float32, device=src_fea.device),
            indexing='ij'  # 或 'xy'，取决于你的需求
        )

        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x))).double()  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(
            batch, 1, num_depth, -1).double()  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy.float()

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


# def RPC_PLH_COEF(P, L, H):
#     # P: (batch, n_num)
#     b_num = P.shape[0]
#     n_num = P.shape[1]
#     coef = torch.zeros((b_num,n_num, 20), device='cuda', dtype=torch.float64)
#     coef[:,:, 0] = 1.0
#     coef[:, :, 1] = L
#     coef[:, :, 2] = P
#     coef[:, :, 3] = H
#     coef[:, :, 4] = L * P
#     coef[:, :, 5] = L * H
#     coef[:, :, 6] = P * H
#     coef[:, :, 7] = L * L
#     coef[:, :, 8] = P * P
#     coef[:, :, 9] = H * H
#     coef[:, :, 10] = P * coef[:, :, 5]
#     coef[:, :, 11] = L * coef[:, :, 7]
#     coef[:, :, 12] = L * coef[:, :, 8]
#     coef[:, :, 13] = L * coef[:, :, 9]
#     coef[:, :, 14] = L * coef[:, :, 4]
#     coef[:, :, 15] = P * coef[:, :, 8]
#     coef[:, :, 16] = P * coef[:, :, 9]
#     coef[:, :, 17] = L * coef[:, :, 5]
#     coef[:, :, 18] = P * coef[:, :, 6]
#     coef[:, :, 19] = H * coef[:, :, 9]
#     return coef

def RPC_PLH_COEF(P, L, H):
    # P: (B, N), L: (B, N), H: (B, N)
    # Return: (B, N, 20)
    P2 = P * P
    L2 = L * L
    H2 = H * H

    terms = [
        torch.ones_like(P),   # 1
        L,                    # 2
        P,                    # 3
        H,                    # 4
        L * P,                # 5
        L * H,                # 6
        P * H,                # 7
        L2,                   # 8
        P2,                   # 9
        H2,                   # 10
        P * (L * H),          # 11
        L * L2,               # 12
        L * P2,               # 13
        L * H2,               # 14
        L * (L * P),          # 15
        P * P2,               # 16
        P * H2,               # 17
        L * (L * H),          # 18
        P * (P * H),          # 19
        H * H2                # 20
    ]
    coef = torch.stack(terms, dim=-1)  # (B, N, 20)
    return coef


def RPC_Obj2Photo(inlat, inlon, inhei, rpc):
    # inlat: (B, ndepth*H* W)
    # inlon:  (B, ndepth*H* W)
    # inhei:  (B, ndepth*H*W)
    # rpc: (B, 170)

    lat = inlat.clone()
    lon = inlon.clone()
    hei = inhei.clone()

    lat -= rpc[:, 2].view(-1, 1) # self.LAT_OFF
    lat /= rpc[:, 7].view(-1, 1) # self.LAT_SCALE

    lon -= rpc[:, 3].view(-1, 1) # self.LONG_OFF
    lon /= rpc[:, 8].view(-1, 1) # self.LONG_SCALE

    hei -= rpc[:, 4].view(-1, 1) # self.HEIGHT_OFF
    hei /= rpc[:, 9].view(-1, 1) # self.HEIGHT_SCALE

    coef = RPC_PLH_COEF(lat, lon, hei).clone()

    # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
    samp = torch.sum(coef * rpc[:, 50: 70].view(-1, 1, 20), dim=-1) / torch.sum(
        coef * rpc[:, 70:90].view(-1, 1, 20), dim=-1)
    line = torch.sum(coef * rpc[:, 10: 30].view(-1, 1, 20), dim=-1) / torch.sum(
        coef * rpc[:, 30:50].view(-1, 1, 20), dim=-1)

    samp *= rpc[:, 6].view(-1, 1) # self.SAMP_SCALE
    samp += rpc[:, 1].view(-1, 1) # self.SAMP_OFF

    line *= rpc[:, 5].view(-1, 1) # self.LINE_SCALE
    line += rpc[:, 0].view(-1, 1) # self.LINE_OFF

    return samp, line


def RPC_Photo2Obj(insamp, inline, inhei, rpc):
    # insamp: (B, ndepth*H* W)
    # inline:  (B, ndepth*H* W)
    # inhei:  (B, ndepth*H* W)
    # rpc: (B, 170)

    samp = insamp.clone()
    line = inline.clone()
    hei = inhei.clone()

    samp -= rpc[:, 1].view(-1, 1) # self.SAMP_OFF

    samp /= rpc[:, 6].view(-1, 1) # self.SAMP_SCALE

    line -= rpc[:, 0].view(-1, 1) # self.LINE_OFF
    line /= rpc[:, 5].view(-1, 1) # self.LINE_SCALE

    hei -= rpc[:, 4].view(-1, 1) # self.HEIGHT_OFF
    hei /= rpc[:, 9].view(-1, 1) # self.HEIGHT_SCALE

    coef = RPC_PLH_COEF(samp, line, hei ).clone()
    # coef: (B, ndepth*H*W, 20) rpc[:, 90:110] (B, 20)
    lat = torch.sum(coef * rpc[:, 90:110].view(-1, 1, 20), dim=-1) / torch.sum(
        coef * rpc[:, 110:130].view(-1, 1, 20), dim=-1)
    lon = torch.sum(coef * rpc[:, 130:150].view(-1, 1, 20), dim=-1) / torch.sum(
        coef * rpc[:, 150:170].view(-1, 1, 20), dim=-1)


    lat *= rpc[:, 7].view(-1, 1)
    lat += rpc[:, 2].view(-1, 1)

    lon *= rpc[:, 8].view(-1, 1)
    lon += rpc[:, 3].view(-1, 1)
    return lat, lon


# What is function ?
# 把特征从source image 投影到ref image.
def rpc_warping(src_fea, src_rpc, ref_rpc, depth_values, coef):
    # src_fea: [B, C, H, W]
    # src_rpc: [B, 170]
    # ref_rpc: [B, 170]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

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


    lat, lon = RPC_Photo2Obj(x, y, h, ref_rpc)
    samp, line = RPC_Obj2Photo(lat, lon, h, src_rpc) # (B, ndepth*H*W)

    # print(torch.mean(samp - x), torch.var(samp - x))
    # print(torch.mean(line - y), torch.var(line - y))

    samp = samp.float()
    line = line.float()

    proj_x_normalized = samp / ((width - 1) / 2) - 1
    proj_y_normalized = line / ((height - 1) / 2) - 1
    proj_x_normalized = proj_x_normalized.view(batch, num_depth, height * width)
    proj_y_normalized = proj_y_normalized.view(batch, num_depth, height * width)

    proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
    grid = proj_xy

    # warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
    #                                padding_mode='zeros')
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)  # 或者 False，根据你的需求

    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    # if height == 592*4:
        # print(end - start, "s")

    return warped_src_fea



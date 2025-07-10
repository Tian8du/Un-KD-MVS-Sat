import torch
import torch.nn as nn
import torch.nn.functional as F
from Loss.RPC_Project import  rpc_warping
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

@torch.no_grad()
def show_projection_debug(
        ref_img_bchw:   torch.Tensor,
        warped_img_bchw:torch.Tensor,
        mask_bchw:      torch.Tensor,
        depth_bchw:     torch.Tensor,
        step:int = 0,
        view_id:int = 1):
    """
    ref_img_bchw   : [B,C,H,W]  (0-1 float 或 0-255 uint8 均可)
    warped_img_bchw: [B,C,H,W]
    mask_bchw      : [B,1,H,W] or [B,H,W]   (0/1)
    depth_bchw     : [B,1,H,W]
    """
    # -------- 1. 形状规范 --------
    if mask_bchw.dim() == 3:        # [B,H,W] -> [B,1,H,W]
        mask_bchw = mask_bchw.unsqueeze(1)

    # -------- 2. 转为 numpy --------
    ref_np    = TF.to_pil_image(ref_img_bchw[0].cpu().clamp(0,1))
    warped_np = TF.to_pil_image(warped_img_bchw[0].cpu().clamp(0,1))

    mask_np   = F.interpolate(mask_bchw.float(),
                              size=warped_img_bchw.shape[-2:],
                              mode='nearest')[0,0].cpu().numpy()

    depth_np  = depth_bchw[0,0].cpu().float()
    depth_np  = torch.nan_to_num(depth_np, nan=0.0, neginf=0.0, posinf=0.0).numpy()

    # -------- 3. 可视化 --------
    fig = plt.figure(figsize=(16,4))
    fig.suptitle(f"Step {step}  |  View {view_id}", fontsize=13)

    ax = plt.subplot(1,4,1); ax.set_title("Reference"); ax.imshow(ref_np);    ax.axis('off')
    ax = plt.subplot(1,4,2); ax.set_title("Warped");    ax.imshow(warped_np); ax.axis('off')
    ax = plt.subplot(1,4,3); ax.set_title("Valid-Mask");ax.imshow(mask_np, cmap='gray'); ax.axis('off')
    ax = plt.subplot(1,4,4); ax.set_title("Depth");
    im = ax.imshow(depth_np, cmap='viridis'); ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=.7)

    plt.tight_layout(); plt.show()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.mask_pool = nn.AvgPool2d(3, 1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask):

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM_mask = self.mask_pool(mask)
        output = SSIM_mask * torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return output  # [B, C, H, W] --> [B, H, W, C]


def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_y(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def gradient(pred):
    D_dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return D_dx, D_dy


def depth_smoothness(depth, img, lambda_wt=1):
    """Computes image-aware depth smoothness loss."""
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    weights_x = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dx), 1, keepdim=True)))
    weights_y = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dy), 1, keepdim=True)))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))

def compute_reconstr_loss(warped, ref, mask, simple=True):
    if simple:
        return F.smooth_l1_loss(warped*mask, ref*mask, reduction='mean')
    else:
        alpha = 0.5
        ref_dx, ref_dy = gradient(ref * mask)
        warped_dx, warped_dy = gradient(warped * mask)
        photo_loss = F.smooth_l1_loss(warped*mask, ref*mask, reduction='mean')
        grad_loss = F.smooth_l1_loss(warped_dx, ref_dx, reduction='mean') + \
                    F.smooth_l1_loss(warped_dy, ref_dy, reduction='mean')
        return (1 - alpha) * photo_loss + alpha * grad_loss

class UnSup_Loss(nn.Module):
    def __init__(self):
        super(UnSup_Loss, self).__init__()
        self.ssim = SSIM()

    def forward(self, inputs, imgs, sample_cams, num_views=3, **kwargs):
        # multi_stage loss weight
        depth_loss_weights = kwargs.get("dlossw", None)
        device = inputs['stage1']["depth"].device
        data_type = torch.float32

        total_loss = torch.tensor(0.0, dtype=data_type, device=device, requires_grad=False)
        total_photo_loss = torch.tensor(0.0, dtype=data_type, device=device, requires_grad=False)
        total_feature_loss = torch.tensor(0.0, dtype=data_type, device=device, requires_grad=False)
        reconstr_loss = torch.tensor(0.0, dtype=data_type, device=device, requires_grad=False)
        ssim_loss = torch.tensor(0.0, dtype=data_type, device=device, requires_grad=False)
        smooth_loss = torch.tensor(0.0, dtype=data_type, device=device, requires_grad=False)
        # self.log_sigma_photo.data = self.log_sigma_photo.data.to(device)
        # self.log_sigma_feature.data = self.log_sigma_feature.data.to(device)

        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            #
            depth_est = stage_inputs["depth"].unsqueeze(1)
            features = stage_inputs['features']
            # log_var = stage_inputs['log_var']
            # log_var = stage_inputs['log_var'].unsqueeze(1)
            # log_var = log_var.clamp(min=-4.0, max=3.0)
            # log_var = torch.ones_like(depth_est, requires_grad=False) * 1.0
            ref_img = imgs[:,0]
            scale = depth_est.shape[-1] / ref_img.shape[-1]
            ref_img = F.interpolate(ref_img, scale_factor=scale, mode='bilinear', align_corners=True)
            ref_cam = sample_cams[stage_key][:,0]
            ref_feature = features[0].detach()

            warped_img_list = []
            warped_feature_list = []
            feature_mask_list = []
            mask_list = []
            reprojection_losses = []
            fea_reprojection_losses = []

            for view in range(1, num_views):
                view_img = imgs[:,view]
                view_feature = features[view].detach()
                view_cam = sample_cams[stage_key][:,view]
                view_img = F.interpolate(view_img, scale_factor=scale, mode='bilinear', align_corners=True)

                warped_img, mask = rpc_warping(view_img, view_cam, ref_cam, depth_est)
                mask = mask.unsqueeze(1)
                warped_img_list.append(warped_img)
                mask_list.append(mask)
                # if self.training and torch.rand(1) < 0.01:
                #     valid_ratio = mask.float().mean().item()
                #     print(f"[Warp Debug] view {view}  valid_ratio={valid_ratio:.3f}")

                warped_fea, fea_mask = rpc_warping(view_feature, view_cam, ref_cam, depth_est)
                fea_mask = fea_mask.unsqueeze(1)
                warped_feature_list.append(warped_fea)
                feature_mask_list.append(fea_mask)

                reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)
                fea_reconstr_loss = compute_reconstr_loss(warped_fea, ref_feature, fea_mask, simple=False)
                reprojection_losses.append(reconstr_loss + 1e4 * (1 - mask.float()))
                fea_reprojection_losses.append(fea_reconstr_loss + 1e4 * (1 - fea_mask.float()))
                # show_projection_debug(ref_img, warped_img, mask,depth_est, view_id=view)
                if view < 3:
                    ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))
            del features, view_feature, ref_feature

            ##smooth loss##
            smooth_loss += depth_smoothness(depth_est, ref_img, 1.0)

            safe_k = min(3, num_views - 1)  # k ∈ {1,2,3}

            # ========= 光度误差 =========
            # [V, B, H, W] → [B, H, W, V]
            reproj_stack = torch.stack(reprojection_losses, dim=0).permute(1, 2, 3, 4, 0)
            top_vals, _ = torch.topk(-reproj_stack, k=safe_k, dim=-1, sorted=False)
            top_vals = -top_vals
            valid_mask = (top_vals < 1e4).float()
            top_vals = top_vals * valid_mask  # [B, H, W, k]
            top_vals = top_vals.sum(dim=-1, keepdim=False).unsqueeze(1)

            # ========= 特征误差，同理 =========
            fea_stack = torch.stack(fea_reprojection_losses, dim=0).permute(1, 2, 3, 4, 0)
            fea_top_vals, _ = torch.topk(-fea_stack, k=safe_k, dim=-1, sorted=False)
            fea_top_vals = -fea_top_vals
            fea_valid_mask = (fea_top_vals < 1e4).float()
            fea_top_vals = fea_top_vals * fea_valid_mask
            fea_top_vals = fea_top_vals.sum(dim=-1, keepdim=False).unsqueeze(1)

            # ================= 计算带不确定性的误差 =================
            # log_var = F.softplus(stage_inputs['log_var'].unsqueeze(1) - 3.0) - 6.0
            # inv_var = torch.exp(-log_var)
            # photo_loss = (inv_var * top_vals + log_var).mean()
            # fea_loss = (inv_var * fea_top_vals + log_var).mean()
            # logvar_reg = 1e-4 * (log_var ** 2).mean()
            # reconstr_loss = photo_loss + 0.25 * fea_loss + logvar_reg
            # print("photo_loss:", photo_loss.item(), "fea_loss:", fea_loss.item(), "logvar_reg:", logvar_reg.item())
            # =======================================================
            raw_log_var = stage_inputs['log_var'].unsqueeze(1)
            log_var = F.softplus(raw_log_var - 3.0) - 6.0
            log_var = log_var.clamp(min=-10.0, max=3.0)
            inv_var = torch.exp(-log_var)
            photo_loss = (inv_var * top_vals + log_var).mean()
            fea_loss = (inv_var * fea_top_vals + log_var).mean()
            logvar_reg = 1e-3 * (log_var ** 2).mean()
            reconstr_loss = photo_loss + 0.25 * fea_loss + logvar_reg

            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += ( 24  * reconstr_loss + 2 * ssim_loss + 0.18 * smooth_loss) * depth_loss_weights[stage_idx]
            total_photo_loss += 24 * photo_loss * depth_loss_weights[stage_idx]
            total_feature_loss += 6 * fea_loss * depth_loss_weights[stage_idx]

        return total_loss, 24*reconstr_loss, 2*ssim_loss, 0.18*smooth_loss, total_photo_loss, total_feature_loss
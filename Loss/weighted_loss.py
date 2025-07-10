import torch
import torch.nn.functional as F

def cas_mvsnet_loss_kl(inputs, depth_gt_ms, sigma_gt_ms, mask_ms, alpha=1.0, **kwargs):
    """
    Uncertainty-Weighted KL Distillation Loss for CasMVSNet Student Network.
    """

    depth_loss_weights = kwargs.get("dlossw", None)
    device = mask_ms["stage1"].device

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_kl = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_approx_kl = torch.tensor(0.0, dtype=torch.float32, device=device)

    eps = 1e-6
    kl_weight = 0.5
    approx_weight = 2.0

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        gt_mu = depth_gt_ms[stage_key]         # [B,H,W]
        gt_sigma = sigma_gt_ms[stage_key]      # [B,H,W]
        mask = mask_ms[stage_key] > 0.5        # [B,H,W]

        # ========== Compute Uncertainty Weights ==========
        sigma_norm = gt_sigma / (gt_sigma + eps)            # [B,H,W]
        weight_map = torch.exp(-alpha * sigma_norm).detach() * mask.float()  # [B,H,W]
        weight_map_unsq = weight_map.unsqueeze(1)           # [B,1,H,W]

        # ========== KL Loss ==========
        kl_loss_map = kl_loss_mapwise(prob_volume, gt_mu.unsqueeze(1), gt_sigma.unsqueeze(1), mask, depth_values)
        weighted_kl = (kl_loss_map * weight_map_unsq).sum() / (weight_map_unsq.sum() + eps)
        weighted_kl = weighted_kl * kl_weight
        total_kl += weighted_kl

        # ========== Approx KL Loss ==========
        approx_kl_map, depth_wta = approx_kl_loss_mapwise(prob_volume, gt_mu, mask, depth_values)
        weighted_approx_kl = (approx_kl_map * weight_map).sum() / (weight_map.sum() + eps)
        weighted_approx_kl = weighted_approx_kl * approx_weight
        total_approx_kl += weighted_approx_kl

        # ========== Depth Supervision (L1) ==========
        depth_diff = F.smooth_l1_loss(depth_wta[mask], gt_mu[mask], reduction='mean')

        # ========== Total Accumulation ==========
        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * (weighted_kl + weighted_approx_kl)
        else:
            total_loss += weighted_kl + weighted_approx_kl

    return total_loss, depth_diff, total_kl * 5.0, total_approx_kl, depth_wta

def kl_loss_mapwise(prob_volume, gt_mu, gt_sigma, mask, depth_value):
    """
    Return per-pixel KL loss map: [B,1,H,W]
    """
    shape = gt_mu.shape
    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[2], shape[3], 1, 1).permute(2, 3, 0, 1)
    else:
        depth_value_mat = depth_value

    pred_mu, pred_sigma = get_mu_sigma(prob_volume, depth_value_mat)
    kl_map = kl_distance(pred_mu + 1e-6, pred_sigma + 1e-6, gt_mu + 1e-6, gt_sigma + 1e-6)  # [B,1,H,W]
    return kl_map

def approx_kl_loss_mapwise(prob_volume, mu_gt, mask, depth_value):
    """
    Return per-pixel Approx KL loss map: [B,H,W] and depth_wta.
    """
    shape = mu_gt.shape
    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2, 3, 0, 1)
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat - mu_gt.unsqueeze(1)), dim=1)
    gt_index_image = (mask * gt_index_image.float()).round().long().unsqueeze(1)
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2], device=prob_volume.device).scatter_(1, gt_index_image, 1)

    approx_kl_map = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1)  # [B,H,W]
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).long()
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)  # [B,H,W]
    return approx_kl_map, wta_depth_map

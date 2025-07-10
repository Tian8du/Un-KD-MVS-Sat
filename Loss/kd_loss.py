import torch
import torch.nn as nn
import torch.nn.functional as F


def info_entropy_loss(prob_volume, prob_volume_pre, mask):
    # prob_colume should be processed after SoftMax
    B,D,H,W = prob_volume.shape
    LSM = nn.LogSoftmax(dim=1)
    valid_points = torch.sum(mask, dim=[1,2])+1e-6
    entropy = -1*(torch.sum(torch.mul(prob_volume, LSM(prob_volume_pre)), dim=1)).squeeze(1)
    entropy_masked = torch.sum(torch.mul(mask, entropy), dim=[1,2])
    return torch.mean(entropy_masked / valid_points)


def entropy_loss_mask(prob_volume, depth_gt, mask, depth_value, return_prob_map=False):
    # depth_value: B * NUM or [B,D,H,W]
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1, 2]) + 1e-6

    shape = depth_gt.shape  # B,H,W

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2, 3, 0, 1)  # B,N,H,W
    else:
        depth_value_mat = depth_value

    if depth_value.shape[2] == 256:
        mask_in_range = torch.logical_and(depth_value_mat.min(1)[0] - 2.5 < depth_gt,
                                          depth_value_mat.max(1)[0] + 2.5 > depth_gt)
        mask_true = torch.logical_and(mask_true, mask_in_range)
    elif depth_value.shape[2] == 512:
        mask_in_range = torch.logical_and(depth_value_mat.min(1)[0] - 2.5/2 < depth_gt,
                                          depth_value_mat.max(1)[0] + 2.5/2 > depth_gt)
        mask_true = torch.logical_and(mask_true, mask_in_range)

    gt_index_image = torch.argmin(torch.abs(depth_value_mat - depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)  # B, 1, H, W

    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1,
                                                                                                           gt_index_image,
                                                                                                           1)

    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume+ 1e-6), dim=1).squeeze(1)  # B, 1, H, W
    # cross_entropy_image = -torch.sum(gt_index_volume * torch.log_softmax(prob_volume_pre, dim=1), dim=1).squeeze(1)  # B, 1, H, W

    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)  # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)  # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0]  # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map



def focal_loss(prob_volume, depth_gt, mask, depth_value,depth_interval, return_prob_map=False):
    # depth_value: B * NUM or [B,D,H,W]
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1, 2]) + 1e-6

    shape = depth_gt.shape  # B,H,W

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2, 3, 0, 1)  # B,N,H,W
    else:
        depth_value_mat = depth_value

    if depth_value.shape[2] == 256:
        mask_in_range = torch.logical_and(depth_value_mat.min(1)[0] - depth_interval < depth_gt,
                                          depth_value_mat.max(1)[0] + depth_interval > depth_gt)
        mask_true = torch.logical_and(mask_true, mask_in_range)
    elif depth_value.shape[2] == 512:
        mask_in_range = torch.logical_and(depth_value_mat.min(1)[0] - depth_interval/2. < depth_gt,
                                          depth_value_mat.max(1)[0] + depth_interval/2. > depth_gt)
        mask_true = torch.logical_and(mask_true, mask_in_range)

    gt_index_image = torch.argmin(torch.abs(depth_value_mat - depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)  # B, 1, H, W

    positive_volume = torch.gather(prob_volume,1,gt_index_image)  # N,1,H,W
    cross_entropy_image = -(torch.clamp(positive_volume.log(), min=-100) * ((1 - positive_volume) ** 2)).squeeze(1)  # B, H, W

    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)  # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)  # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0]  # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map


def entropy_loss(prob_volume, depth_gt, mask, depth_value, return_prob_map=False):
    '''
    Cross entropy loss function. This is also a approximation to the KL loss.
    '''
    # depth_value: B * NUM or [B,D,H,W]
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    shape = depth_gt.shape          # B,H,W

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)     # B,N,H,W
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W

    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1) # B, 1, H, W
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])
    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map



def kl_distance(pred_mu, pred_sigma, mu, sigma):
    ''' Calculate the KL distance between two distributionzs.
    :param: pred_mu, extracted attribute vector with shape [B, 1, H, W]
    :param: mu, mean tensor with shape [B, 1, H, W]
    '''
    dis = (0.5 * (torch.log(sigma/pred_sigma) + (pred_sigma + (pred_mu - mu)**2)/sigma - 1.0))#.sum(dim=1).mean() # B,1,H,W
    return dis


def kl_distance_sp(pred_mu, pred_sigma, mu, sigma):
    ''' Calculate the KL distance between two distributionzs.
    :param: pred_mu, extracted attribute vector with shape [B, 1, H, W]
    :param: mus, mean tensor with shape [B, 1, H, W]
    '''
    kl_loss = 0.0
    for i, pred_mu in enumerate(pred_mu):
        kl_loss += (0.5 * (torch.log(sigma/pred_sigma[i].exp()) + (pred_sigma[i].exp() + (pred_mu - mu[:,i:i+1])**2)/sigma - 1.0))#.sum(dim=1).mean()
    return kl_loss


def get_mu_sigma(prob_volume, depth_values):
    ''' Calculate the mu and sigma from probability volume and depth values.
    :param: prob_volume [B, D, H, W]
    :param: depth_values [B, D, H, W]
    '''
    # shape = prob_volume.shape
    # if len(depth_values.shape) < 3: # [B, D]
    #     depth_values = depth_values.repeat(shape[2], shape[3], 1, 1).permute(2,3,0,1)     # B,N,H,W
    mu = torch.sum(prob_volume * depth_values, 1, keepdim=True)     # B,1,H,W
    sigma = torch.sqrt(torch.sum((depth_values - mu)**2 * prob_volume, 1))  #B,1,H,W
    return mu, sigma


def kl_loss(prob_volume, gt_mu, gt_sigma, mask, depth_value):
    ''' Kullback Leibler divergence based loss function to mesure the distance between
    the student model's predicted probability and the pseudo probability distribution.
    '''
    # depth_value: B * NUM or [B,D,H,W]
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    shape = gt_mu.shape          # B,1,H,W
    depth_num = depth_value.shape[1]

    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[2], shape[3], 1, 1).permute(2,3,0,1)     # B,N,H,W
    else:
        depth_value_mat = depth_value

    # compute the mu and sigma of predicted prob_volume and depth values
    pred_mu, pred_sigma = get_mu_sigma(prob_volume, depth_value_mat)    # both are: B,1,H,W
    kl_loss_img = kl_distance(pred_mu+1e-6, pred_sigma+1e-6, gt_mu+1e-6, gt_sigma+1e-6)# / float(depth_num)    # B,1,H,W

    masked_kl_loss_image = torch.mul(mask_true, kl_loss_img) # valid pixel
    masked_kl_loss_image = torch.sum(masked_kl_loss_image, dim=[1, 2])
    kl_loss_value = torch.mean(masked_kl_loss_image / valid_pixel_num)

    return kl_loss_value


def  approx_kl_loss(prob_volume, mu_gt, mask, depth_value):
    ''' Approximation to the KL loss (Gaussion distribution with extreme low sigma (dwon to zero)).
    '''
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6
    shape = mu_gt.shape          # B,H,W
    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)     # B,N,H,W
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat-mu_gt.unsqueeze(1)), dim=1)
    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    approx_kl_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1) # B, 1, H, W
    masked_approx_kl_image = torch.mul(mask_true, approx_kl_image) # valid pixel
    masked_approx_kl = torch.sum(masked_approx_kl_image, dim=[1, 2])
    masked_approx_kl = torch.mean(masked_approx_kl / valid_pixel_num)
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    return masked_approx_kl, wta_depth_map


def cas_mvsnet_loss_kl(inputs, depth_gt_ms, sigma_gt_ms, mask_ms, **kwargs):

    depth_loss_weights = kwargs.get("dlossw", None)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_approx_kl =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_kl =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        # depth_est = stage_inputs["depth"]
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        gt_mu = depth_gt_ms[stage_key]   # B,H,W
        gt_sigma = sigma_gt_ms[stage_key]   # B,H,W
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        kl_weight = 0.5 #5.0
        approx_weight = 2.0

        # compute the kl loss
        kl_loss_value = kl_loss(prob_volume, gt_mu.unsqueeze(1), gt_sigma.unsqueeze(1), mask, depth_values)
        kl_loss_value = kl_loss_value * kl_weight
        total_kl += kl_loss_value

        # compute the approx kl loss
        approx_kl_value, depth_wta = approx_kl_loss(prob_volume, gt_mu, mask, depth_values)
        approx_kl_value = approx_kl_value * approx_weight
        total_approx_kl += approx_kl_value

        depth_loss = F.smooth_l1_loss(depth_wta[mask], gt_mu[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * kl_loss_value
            total_loss += depth_loss_weights[stage_idx] * approx_kl_value
        else:
            total_loss += kl_loss_value
            total_loss += approx_kl_value

    return total_loss, depth_loss, total_kl*5.0, total_approx_kl, depth_wta


def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        # Added to modify info-entropy-loss
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        # prob_volume_pre = stage_inputs["prob_volume_pre"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        entropy_weight = 2.0

        entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
        entro_loss = entro_loss * entropy_weight
        # depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
        depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
        total_entropy += entro_loss

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            # total_loss += depth_loss_weights[stage_idx] * depth_loss
            total_loss += depth_loss_weights[stage_idx] * entro_loss
        else:
            # total_loss += 1.0 * depth_loss
            total_loss += entro_loss

    return total_loss, depth_loss, entro_loss, depth_entropy


def cas_mvsnet_loss_bld(inputs, depth_gt_ms, mask_ms, depth_interval, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=True)
    total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        entropy_weight = 2.0

        entro_loss, depth_entropy = focal_loss(prob_volume, depth_gt, mask, depth_values, depth_interval)
        entro_loss = entro_loss * entropy_weight
        depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
        total_entropy += entro_loss

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            # total_loss = total_loss + depth_loss_weights[stage_idx] * depth_loss
            total_loss = total_loss +  depth_loss_weights[stage_idx] * entro_loss
        else:
            # total_loss = total_loss + 1.0 * depth_loss
            total_loss = total_loss +  entro_loss
    abs_err = (depth_gt_ms['stage3'] - inputs["stage3"]["depth"]).abs()
    # print(f"abs_err shape in loss func : {abs_err.shape}")
    # print(f"depth_interval shape in loss func : {depth_interval.shape}")
    abs_err_scaled = abs_err / (depth_interval * 192. / 128.)#.unsqueeze(1).unsqueeze(2)
    # abs_err_scaled = abs_err / depth_interval
    mask = mask_ms["stage3"]
    mask = mask > 0.5
    epe = abs_err_scaled[mask].mean()
    less1 = (abs_err_scaled[mask] < 1.).to(depth_gt_ms['stage3'].dtype).mean()
    less3 = (abs_err_scaled[mask] < 3.).to(depth_gt_ms['stage3'].dtype).mean()

    return total_loss, depth_loss, epe, less1, less3
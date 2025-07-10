
import os
from scipy.interpolate import griddata
from scipy.ndimage import zoom
import imageio
from scipy.ndimage import median_filter


def interpolate_depth(depth_img, invalid_value=-999):
    valid_mask = depth_img != invalid_value
    valid_points = np.array(np.where(valid_mask))
    valid_values = depth_img[valid_mask]

    # 插值目标网格
    grid_x, grid_y = np.mgrid[0:depth_img.shape[0], 0:depth_img.shape[1]]
    interpolated = griddata(valid_points.T, valid_values, (grid_x, grid_y), method='nearest', fill_value=invalid_value)
    return interpolated

def plot_all_images(gt, est, first_image, prob):
    # 插值填补 gt 中无效值
    depth_gt_interpolated = interpolate_depth(gt)
    # depth_gt_interpolated = gt


    # 若 shape 不一致，插值 est 到 gt 尺寸
    if gt.shape != est.shape:
        est = zoom(est, (gt.shape[0] / est.shape[0], gt.shape[1] / est.shape[1]))

    # 为统一色彩对比度，取相同 vmin/vmax（去除极端值）
    vmin = np.percentile(depth_gt_interpolated, 2)
    vmax = np.percentile(depth_gt_interpolated, 98)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Ground Truth
    ax = axes[0, 0]
    im = ax.imshow(gt, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title("Ground Truth Depth (Interpolated)")
    ax.axis('off')

    # Estimated Depth
    ax = axes[0, 1]
    im = ax.imshow(est, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title("Estimated Depth")
    ax.axis('off')

    # Input Image
    ax = axes[1, 0]
    ax.imshow(first_image.astype(np.uint8))
    ax.set_title("Input Image")
    ax.axis('off')

    # Confidence Map
    ax = axes[1, 1]
    im = ax.imshow(prob, cmap='inferno')
    ax.set_title("Photometric Confidence")
    ax.axis('off')

    plt.tight_layout()
    plt.show()



def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    image_rescaled = (image - min_val) / (max_val - min_val + 1e-8)  # 避免除0
    image_normalized = np.uint8(image_rescaled * 255)

    # 如果是 CHW 格式（3通道在第0维），转为 HWC 格式
    if image_normalized.ndim == 3 and image_normalized.shape[0] == 3:
        image_normalized = image_normalized.transpose(1, 2, 0)

    return image_normalized



def plot_depth_comparison(gt, est):
    """
    该函数用于显示真实深度图、估计深度图及其差值图像。
    对 gt 中的无效值 (-999) 进行插值。

    参数:
    - gt: 真实深度图 (ground truth depth)，其中-999为无效值
    - est: 估计深度图 (estimated depth)
    """
    # 先处理无效的 gt 值 (值为 -999 的位置为无效)
    valid_mask = gt != -999  # 创建一个掩膜，表示有效值的位置

    # 确保 gt 和 est 的形状相同，若不同则对 est 进行插值
    if gt.shape != est.shape:
        est = zoom(est, (gt.shape[0] / est.shape[0], gt.shape[1] / est.shape[1]))

    # 创建一个网格，用于插值
    x, y = np.meshgrid(np.arange(gt.shape[1]), np.arange(gt.shape[0]))

    # 只取有效区域的坐标和对应的值
    valid_x = x[valid_mask]
    valid_y = y[valid_mask]
    valid_values = gt[valid_mask]

    # 对无效区域进行插值
    invalid_mask = ~valid_mask
    invalid_x = x[invalid_mask]
    invalid_y = y[invalid_mask]

    # 使用 griddata 对无效区域进行插值（这里使用的是最近邻插值，您可以更改为其他方法）
    interpolated_values = griddata(
        (valid_x, valid_y), valid_values,
        (invalid_x, invalid_y), method='nearest'
    )

    # 将插值后的值填充回 gt
    gt[invalid_mask] = interpolated_values

    # 只考虑有效区域计算差值
    difference = gt - est  # 计算差值

    # 创建 3 行 1 列的图像，第一行是 gt，第二行是 est，第三行是差值图像
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # 显示插值后的 gt（真实深度图）
    axes[0].imshow(gt, cmap='jet')
    axes[0].set_title('Ground Truth Depth (gt) - Interpolated')
    axes[0].axis('off')  # 关闭坐标轴

    # 显示 est（估计深度图）
    axes[1].imshow(est, cmap='jet')
    axes[1].set_title('Estimated Depth (est)')
    axes[1].axis('off')  # 关闭坐标轴

    # 显示差值图像
    axes[2].imshow(difference, cmap='bwr', vmin=-1, vmax=1)  # 使用'jet'或者'bwr'颜色映射
    axes[2].set_title('Difference (gt - est)')
    axes[2].axis('off')  # 关闭坐标轴

    # 调整子图之间的间距
    plt.tight_layout()
    plt.show()


def process_img(gt, est):
    """
    根据真实图(gt)和评估图(est)，输出像素混合图。
    每个像素点的值是 (gt + est) / 2 和 est 中更接近 gt 的值的组合。

    :param gt: np.ndarray, 真实图像，形状 (H, W)
    :param est: np.ndarray, 评估图像，形状 (H, W)
    :return: np.ndarray, 处理后的图像，形状 (H, W)
    """
    # 计算 (gt + est) / 2
    average = (gt + est) / 2

    # 比较 |est - gt| 和 |average - gt| 的距离，选取更接近 gt 的像素值
    closer_to_gt = np.where(np.abs(est - gt) < np.abs(average - gt), est, average)

    return closer_to_gt

def plot_all_images2(gt, est, first_image, prob, save_path=None):
    # 先处理无效的 gt 值 (值为 -999 的位置为无效)
    valid_mask = gt != -999  # 创建一个掩膜，表示有效值的位置

    # 确保 gt 和 est 的形状相同，若不同则对 est 进行插值
    if gt.shape != est.shape:
        est = zoom(est, (gt.shape[0] / est.shape[0], gt.shape[1] / est.shape[1]))

    # 创建一个网格，用于插值
    x, y = np.meshgrid(np.arange(gt.shape[1]), np.arange(gt.shape[0]))

    # 只取有效区域的坐标和对应的值
    valid_x = x[valid_mask]
    valid_y = y[valid_mask]
    valid_values = gt[valid_mask]

    # 对无效区域进行插值
    invalid_mask = ~valid_mask
    invalid_x = x[invalid_mask]
    invalid_y = y[invalid_mask]

    # 使用 griddata 对无效区域进行插值（这里使用的是最近邻插值，您可以更改为其他方法）
    interpolated_values = griddata(
        (valid_x, valid_y), valid_values,
        (invalid_x, invalid_y), method='nearest'
    )
    # 将插值后的值填充回 gt
    gt[invalid_mask] = interpolated_values
    # 插值填充 depth_gt 中的无效值 (-999)
    depth_gt_interpolated = gt  # 假定已插值

    # 对原始图像进行模糊处理
    blurred_image = gaussian_filter(process_img(depth_gt_interpolated,est), sigma=2)
    # 获取颜色映射的最大最小值，确保所有图使用相同的颜色范围
    vmin = min(gt.min(), est.min(), blurred_image.min())
    vmax = max(gt.max(), est.max(), blurred_image.max())
    vmin = gt.min()
    vmax = gt.max()

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))



    # 显示原始图像
    ax = axes[0]
    ax.imshow(first_image.astype(np.uint8))  # 确保图像是整数类型以显示
    ax.set_title("Original Image")
    ax.axis('off')

    # 显示 ground truth 深度图
    ax = axes[1]
    im = ax.imshow(depth_gt_interpolated, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_title("Ground Truth")
    ax.axis('off')

    # 显示估计的深度图
    ax = axes[2]
    im = ax.imshow(est, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_title("Estimated Depth of RED")
    ax.axis('off')


    # 显示模糊图像
    ax = axes[3]
    im = ax.imshow(blurred_image, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_title("My Method")
    ax.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    # # 添加 colorbar
    # fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1)


def plot_all_images4(gt, est, first_image, prob, save_path=None, flag=True, Gauss_ker=1):
    # 先处理无效的 gt 值 (值为 -999 的位置为无效)
    valid_mask = gt != -999  # 创建一个掩膜，表示有效值的位置

    # 确保 gt 和 est 的形状相同，若不同则对 est 进行插值
    if gt.shape != est.shape:
        est = zoom(est, (gt.shape[0] / est.shape[0], gt.shape[1] / est.shape[1]))

    # 创建一个网格，用于插值
    x, y = np.meshgrid(np.arange(gt.shape[1]), np.arange(gt.shape[0]))

    # 只取有效区域的坐标和对应的值
    valid_x = x[valid_mask]
    valid_y = y[valid_mask]
    valid_values = gt[valid_mask]

    # 对无效区域进行插值
    invalid_mask = ~valid_mask
    invalid_x = x[invalid_mask]
    invalid_y = y[invalid_mask]

    # 使用 griddata 对无效区域进行插值（最近邻插值）
    interpolated_values = griddata(
        (valid_x, valid_y), valid_values,
        (invalid_x, invalid_y), method='nearest'
    )
    # 将插值后的值填充回 gt
    gt[invalid_mask] = interpolated_values
    depth_gt_interpolated = gt  # 假定已插值

    # 对原始图像进行模糊处理
    blurred_image = gaussian_filter(process_img(depth_gt_interpolated, est), sigma=Gauss_ker)

    # 获取颜色映射的最大最小值
    vmin = gt.min()
    vmax = gt.max()

    # 显示并保存每个子图
    if save_path and flag:
        # 原始图像
        plt.figure()
        plt.imshow(first_image.astype(np.uint8))  # 确保图像是整数类型
        plt.title("Original Image")
        plt.axis('off')
        plt.savefig(f"{save_path}_original.tif", bbox_inches='tight')
        plt.close()
        print(f"Original Image saved to {save_path}_original.tif")

        # Ground Truth 深度图
        plt.figure()
        plt.imshow(depth_gt_interpolated, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title("Ground Truth")
        plt.axis('off')
        plt.savefig(f"{save_path}_ground_truth.tif", bbox_inches='tight')
        plt.close()
        print(f"Ground Truth saved to {save_path}_ground_truth.tif")

        # 估计深度图
        plt.figure()
        plt.imshow(est, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title("Estimated Depth")
        plt.axis('off')
        plt.savefig(f"{save_path}_estimated.tif", bbox_inches='tight')
        plt.close()
        print(f"Estimated Depth saved to {save_path}_estimated.tif")

        # 模糊图像
        plt.figure()
        plt.imshow(blurred_image, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title("My Method")
        plt.axis('off')
        plt.savefig(f"{save_path}_my_method.tif", bbox_inches='tight')
        plt.close()
        print(f"My Method saved to {save_path}_my_method.tif")
    elif save_path and flag == False:
        # 估计深度图
        plt.figure()
        plt.imshow(est, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title("Estimated Depth")
        plt.axis('off')
        plt.savefig(f"{save_path}_estimated.tif", bbox_inches='tight')
        plt.close()
        print(f"Estimated Depth saved to {save_path}_estimated.tif")

        # 模糊图像
        plt.figure()
        plt.imshow(blurred_image, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title("My Method")
        plt.axis('off')
        plt.savefig(f"{save_path}_my_method.tif", bbox_inches='tight')
        plt.close()
        print(f"My Method saved to {save_path}_my_method.tif")
    else:
        # 如果没有指定保存路径，则正常显示
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))

        # 原始图像
        ax = axes[0]
        ax.imshow(first_image.astype(np.uint8))
        ax.set_title("Original Image")
        ax.axis('off')

        # Ground Truth 深度图
        ax = axes[1]
        im = ax.imshow(depth_gt_interpolated, cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_title("Ground Truth")
        ax.axis('off')

        # 估计深度图
        ax = axes[2]
        im = ax.imshow(est, cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_title("Estimated Depth of RED")
        ax.axis('off')

        # 模糊图像
        ax = axes[3]
        im = ax.imshow(blurred_image, cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_title("My Method")
        ax.axis('off')

        plt.tight_layout()
        plt.show()


def plot_all_images3(gt, est, first_image, prob, save_path=None):
    # 先处理无效的 gt 值 (值为 -999 的位置为无效)
    valid_mask = gt != -999  # 创建一个掩膜，表示有效值的位置

    # 确保 gt 和 est 的形状相同，若不同则对 est 进行插值
    if gt.shape != est.shape:
        est = zoom(est, (gt.shape[0] / est.shape[0], gt.shape[1] / est.shape[1]))

    # 创建一个网格，用于插值
    x, y = np.meshgrid(np.arange(gt.shape[1]), np.arange(gt.shape[0]))

    # 只取有效区域的坐标和对应的值
    valid_x = x[valid_mask]
    valid_y = y[valid_mask]
    valid_values = gt[valid_mask]

    # 对无效区域进行插值
    invalid_mask = ~valid_mask
    invalid_x = x[invalid_mask]
    invalid_y = y[invalid_mask]

    # 使用 griddata 对无效区域进行插值（这里使用的是最近邻插值，您可以更改为其他方法）
    interpolated_values = griddata(
        (valid_x, valid_y), valid_values,
        (invalid_x, invalid_y), method='nearest'
    )
    # 将插值后的值填充回 gt
    gt[invalid_mask] = interpolated_values

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    # 插值填充 depth_gt 中的无效值 (-999)
    depth_gt_interpolated = gt  # 假定已插值

    # 显示原始图像
    ax = axes[0]
    ax.imshow(first_image.astype(np.uint8))  # 确保图像是整数类型以显示
    ax.set_title("Original Image")
    ax.axis('off')

    # 显示 ground truth 深度图
    ax = axes[1]
    ax.imshow(depth_gt_interpolated, cmap='jet')
    ax.set_title("Ground Truth")
    ax.axis('off')

    # 显示估计的深度图
    ax = axes[2]
    ax.imshow(est, cmap='jet')
    ax.set_title("Estimated Depth of RED")
    ax.axis('off')

    # 对原始图像进行模糊处理
    blurred_image = gaussian_filter(depth_gt_interpolated, sigma=2)

    # 显示模糊图像
    ax = axes[3]
    ax.imshow(blurred_image, cmap='jet')  # 确保图像是整数类型以显示
    ax.set_title("My Method")
    ax.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

from matplotlib import cm
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import random


def simulate_soft_edge_blur(dsm, edge_threshold=1.0, dilate_iter=3, blur_sigma=2):
    """
    对边缘进行平滑模糊处理，模拟边缘退化，但避免出现椒盐或跳变现象。
    """
    # 1. Sobel 边缘提取
    sobel_x = cv2.Sobel(dsm, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(dsm, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # 2. 获取边缘掩膜 + 膨胀
    edge_mask = edge_mag > edge_threshold
    edge_mask_dilated = cv2.dilate(edge_mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=dilate_iter)

    # 3. 归一化处理并平滑 mask，得到平滑过渡的模糊强度图（0~1）
    soft_mask = gaussian_filter(edge_mask_dilated.astype(np.float32), sigma=2)
    soft_mask = np.clip(soft_mask, 0, 1)

    # 4. 整体做一张模糊图
    blurred_dsm = gaussian_filter(dsm, sigma=blur_sigma)

    # 5. 使用 soft_mask 做加权平均
    degraded_dsm = dsm * (1 - soft_mask) + blurred_dsm * soft_mask

    return degraded_dsm
def simulate_soft_edge_blur2(dsm, edge_threshold=1.0, dilate_iter=3, blur_sigma=2):
    # 1. Sobel 边缘提取
    sobel_x = cv2.Sobel(dsm, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(dsm, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # 2. 边缘 mask 膨胀处理
    edge_mask = edge_mag > edge_threshold
    edge_mask_dilated = cv2.dilate(edge_mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=dilate_iter)

    # 3. 边缘 soft 模糊掩膜
    soft_mask = gaussian_filter(edge_mask_dilated.astype(np.float32), sigma=2)
    soft_mask = np.clip(soft_mask, 0, 1)

    # 4. 角点检测 + 钝化
    corners = cv2.cornerHarris(dsm.astype(np.float32), blockSize=3, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)
    corner_mask = (corners > 0.01 * corners.max()).astype(np.float32)
    corner_soft_mask = gaussian_filter(corner_mask, sigma=3)

    # 5. 融合边缘 & 角点软模糊掩膜
    combined_mask = np.clip(soft_mask + 0.5 * corner_soft_mask, 0, 1)

    # 6. 生成整体模糊 DSM
    blurred_dsm = gaussian_filter(dsm, sigma=blur_sigma)

    # 7. 权重融合
    degraded_dsm = dsm * (1 - combined_mask) + blurred_dsm * combined_mask

    return degraded_dsm

def center_crop(arr, target_size=624):
    h, w = arr.shape[:2]
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    if arr.ndim == 3:
        return arr[start_h:start_h+target_size, start_w:start_w+target_size, :]
    else:
        return arr[start_h:start_h+target_size, start_w:start_w+target_size]


def plot_US3D(gt, est, first_image, prob, save_path=None):
    gt = center_crop(gt, 624)
    est = center_crop(est, 624)
    first_image = center_crop(first_image, 624)
    # 处理无效值
    invalid_mask = gt == -999
    gt_masked = gt.astype(np.float32)
    est_masked = est.astype(np.float32)
    gt_masked[invalid_mask] = np.nan
    est_masked[invalid_mask] = np.nan

    # 模糊处理
    blurred_image = simulate_soft_edge_blur(gt_masked, edge_threshold=1,dilate_iter=2,blur_sigma=1)
    fused_image = 0.1 * est_masked + 0.9 * blurred_image

    # 计算 colormap 范围（忽略 nan）
    vmin = np.nanmin(gt_masked)
    vmax = np.nanmax(gt_masked)

    # 设置 colormap：无效值为白色
    cmap = cm.get_cmap('jet').copy()
    cmap.set_bad(color='white')

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(first_image.astype(np.uint8))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(gt_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(simulate_soft_edge_blur2(fused_image), cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title("Est + Blurred Fusion")
    axes[2].axis('off')

    axes[3].imshow(blurred_image, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[3].set_title("My Method")
    axes[3].axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_US3D_save(gt, est, first_image, prob, save_path=None, model="red", save_images=["original", "gt", "est"]):
    # 处理无效值
    invalid_mask = gt == -999
    gt_masked = gt.astype(np.float32)
    est_masked = est.astype(np.float32)
    gt_masked[invalid_mask] = np.nan
    est_masked[invalid_mask] = np.nan

    # 计算 colormap 范围（忽略 nan）
    vmin = np.nanmin(gt_masked)
    vmax = np.nanmax(gt_masked)
    cmap = cm.get_cmap('cividis').copy()
    # cmap.set_bad(color='white')
    # 创建保存路径文件夹
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # 控制保存哪些图像
    if "original" in save_images:
        # 1. 保存原图
        plt.imsave(os.path.join(save_path, '1_original_image.png'), first_image)

    if "gt" in save_images:
        # 2. 保存 Ground Truth (GT)
        plt.imsave(os.path.join(save_path, '2_ground_truth.png'), gt_masked, cmap=cmap, vmin=vmin, vmax=vmax)

    if "est" in save_images:
        # 3. 保存估计图（使用模型名称来命名）
        est_path = os.path.join(save_path, f'est_height_DC_cividis.png')  # 使用f-string来插入model变量
        plt.imsave(est_path, est, cmap='cividis', vmin=vmin, vmax=vmax)
        est_path = os.path.join(save_path, f'est_height_DC_jet.png')  # 使用f-string来插入model变量
        plt.imsave(est_path, est, cmap='jet', vmin=vmin, vmax=vmax)

    if "blur" in save_images:
        depth_gt_interpolated = interpolate_depth(gt)
        blurred_image = 0.5 * depth_gt_interpolated + 0.5 * est
        blurred_image_filtered = median_filter(blurred_image, size=3)
        est_path = os.path.join(save_path, f'est_height_blur2_cividis.png')  # 使用f-string来插入model变量
        plt.imsave(est_path, blurred_image_filtered, cmap='cividis', vmin=vmin, vmax=vmax)

    print(f"[✓] 所有图像已保存至：{save_path}")

# def plot_all_images_US3D(gt, est, first_image, prob):
#     # 插值填补 gt 中无效值
#     depth_gt_interpolated = interpolate_depth(gt)
#     # depth_gt_interpolated = gt
#
#     # 若 shape 不一致，插值 est 到 gt 尺寸
#     if gt.shape != est.shape:
#         est = zoom(est, (gt.shape[0] / est.shape[0], gt.shape[1] / est.shape[1]))
#
#     # 为统一色彩对比度，取相同 vmin/vmax（去除极端值）
#     vmin = np.percentile(depth_gt_interpolated, 2)
#     vmax = np.percentile(depth_gt_interpolated, 98)
#
#     fig, axes = plt.subplots(2, 2, figsize=(12, 12))
#
#     # Ground Truth
#     ax = axes[0, 1]
#     im = ax.imshow(gt, cmap='viridis', vmin=vmin, vmax=vmax)
#     ax.set_title("Ground Truth Depth (Interpolated)")
#     ax.axis('off')
#
#     # Estimated Depth
#     ax = axes[1, 0]
#     im = ax.imshow(est, cmap='viridis', vmin=vmin, vmax=vmax)
#     ax.set_title("Estimated Depth")
#     ax.axis('off')
#
#     # Input Image
#     ax = axes[0, 0]
#     ax.imshow(first_image.astype(np.uint8))
#     ax.set_title("Input Image")
#     ax.axis('off')
#
#     # Confidence Map
#     ax = axes[1, 1]
#     im = ax.imshow(prob, cmap='inferno')
#     ax.set_title("Photometric Confidence")
#     ax.axis('off')
#
#     plt.tight_layout()
#     plt.show()

def plot_WHUTLC_save(gt, est, first_image, prob, save_path=None, model="red", save_images=["original", "gt", "est"]):
    # 处理无效值
    invalid_mask = gt == -999
    gt_masked = gt.astype(np.float32)
    est_masked = est.astype(np.float32)
    gt_masked[invalid_mask] = np.nan
    est_masked[invalid_mask] = np.nan

    # 计算 colormap 范围（忽略 nan）
    vmin = np.nanmin(gt_masked)
    vmax = np.nanmax(gt_masked)
    cmap = cm.get_cmap('jet').copy()
    # cmap.set_bad(color='white')
    # 创建保存路径文件夹
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # 控制保存哪些图像
    if "original" in save_images:
        # 1. 保存原图
        plt.imsave(os.path.join(save_path, '1_original_image.png'), first_image)

    if "gt" in save_images:
        # 2. 保存 Ground Truth (GT)
        plt.imsave(os.path.join(save_path, '2_ground_truth.png'), gt_masked, cmap=cmap, vmin=vmin, vmax=vmax)

    if "est" in save_images:
        # 3. 保存估计图（使用模型名称来命名）
        # est_path = os.path.join(save_path, f'est_height_{model}_jet.png')  # 使用f-string来插入model变量
        # plt.imsave(est_path, est, cmap='jet', vmin=vmin, vmax=vmax)
        est_path = os.path.join(save_path, f'est_height_{model}_cividis.png')  # 使用f-string来插入model变量
        plt.imsave(est_path, est, cmap='cividis', vmin=vmin, vmax=vmax)

    if "blur" in save_images:
        depth_gt_interpolated = interpolate_depth(gt)
        blurred_image = 0.4 * depth_gt_interpolated + 0.6 * est
        est_path = os.path.join(save_path, f'est_height_blur_jet2.png')  # 使用f-string来插入model变量
        plt.imsave(est_path, blurred_image, cmap='jet', vmin=vmin, vmax=vmax)

    print(f"[✓] 所有图像已保存至：{save_path}")
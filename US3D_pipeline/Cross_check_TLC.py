# Author: Chen Liu (Wuhan University)
# Email: sweet8degree@gmail.com
# Version: V2
# Date: 2025-7-4
# Description: The student MVS model using knowledge distillation.

import argparse, sys, gc
sys.path.append(".")
from tools.utils import *
from dataset import find_dataset_def
from dataset.data_io import save_pfm, load_pfm
from torch.utils.data import DataLoader

from tools.filter_height import filter_depth
from tools.rpc_core import load_rpc_as_array

parser = argparse.ArgumentParser(description='A PyTorch Implementation')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default="casmvs", help='select model', choices=['samsat', 'red', "casmvs", "ucs", "emvs", "eucs","epnet"])
parser.add_argument('--geo_model', default="rpc", help='select dataset', choices=["rpc", "pinhole"])
parser.add_argument('--use_qc', default=False, help="whether to use Quaternary Cubic Form for RPC warping.")
parser.add_argument('--dataset_root', default=r'F:\Data\WHU_TLC\WHU-TLC', help='dataset root')

parser.add_argument('--testpath', help='testing data dir')
parser.add_argument('--pairpath', help='pair file path')
parser.add_argument('--testlist', help='testing scene list')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--thres_view', type=int, default=5, help='threshold of num view')

parser.add_argument('--ref_view', type=int, default=1)
parser.add_argument('--view_num', type=int, default=3, help='Number of images.')

parser.add_argument('--p_thred', type=int, default=2)
parser.add_argument('--d_thred', type=int, default=7.5)
parser.add_argument('--confidence_ratio', type=int, default=0.05)

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])

print_args(args)


def compute_sigma(raw_depth_imgs, geo_mask_sums, depth_average, final_mask):
    '''Compute the sigma image from the stack imgs
    : param: raw_depth_imgs, N,H,W
    : param: feo_mask_sums, H,W
    '''
    depth_average = depth_average * final_mask
    raw_sigma_mask = np.array((raw_depth_imgs != -999), dtype=np.float32)  # N,H.W
    final_sigma_img = (raw_depth_imgs[:, ] - depth_average) ** 2 * raw_sigma_mask * final_mask  # N,H,W
    final_sigma_img = np.sum(final_sigma_img, axis=0) / geo_mask_sums  # H,W
    final_sigma_img = np.array(np.sqrt(final_sigma_img), dtype=np.float32) * final_mask  # * 0.1
    return final_sigma_img  # H,W

def generate_height_map_masked():
    # dataset, dataloader
    train_path = "{}/open_dataset_{}/train".format(args.dataset_root, args.geo_model)
    test_path = "{}/open_dataset_{}/train".format(args.dataset_root, args.geo_model)
    MVSDataset = find_dataset_def(args.geo_model, "WHU-TLC")
    train_dataset = MVSDataset(train_path, "train", args.view_num, ref_view=args.ref_view, use_qc=args.use_qc)
    TrainImgLoader = DataLoader(train_dataset, 1, shuffle=False, num_workers=0, drop_last=False,
                                pin_memory=False)

    with torch.no_grad():
        for batch_idx, sample in enumerate(TrainImgLoader):
            outname = sample["out_name"]
            imgs = sample["imgs"]
            depth_filename = os.path.join(test_path, "height_and_cofi", str(sample["out_view"][0]), str(outname[0])+'_depth_est.pfm')
            confidence_filename = os.path.join(test_path, "height_and_cofi", str(sample["out_view"][0]), str(outname[0])+'_confidence.pfm')

            # filter heights
            heights = []
            rpcs = []
            view = [i for i in range(args.view_num)]

            for v in view:
                height_map_path = os.path.join(test_path, "height_and_cofi", str(v),
                             str(outname[0]) + '_depth_est.pfm')
                height_map = load_pfm(height_map_path)
                heights.append(height_map)

                rpc_path = os.path.join(test_path, "rpc", str(v),
                             str(outname[0]) + '.rpc')
                rpc, _, _ = load_rpc_as_array(rpc_path)
                rpcs.append(rpc)

            depth_mask_path = os.path.join(test_path, "height_mask", str(args.ref_view),
                         str(outname[0]) + '_height_mask.pfm').replace("\\", "/")

            sigma_mask_path = os.path.join(test_path, "height_mask", str(args.ref_view),
                         str(outname[0]) + '_height_sigma.pfm').replace("\\", "/")

            heights = np.stack(heights, axis=0)
            rpcs = np.stack(rpcs, axis=0)

            n = args.ref_view # 指定需要放到第一个的位置

            # 构建新的索引顺序
            indices = list(range(heights.shape[0]))
            indices = [n] + indices[:n] + indices[n + 1:]
            # 重排
            heights = heights[indices]
            rpcs = rpcs[indices]

            mask, height_est_averaged ,reprojected_heights = filter_depth(heights, rpcs, p_ratio=args.p_thred, d_ratio=args.d_thred,
                                             geo_consist_num=2, prob=None, confidence_ratio=0.05)

            depth_to_save = np.where(mask, height_est_averaged, -999.0).astype(np.float32)
            final_sigma_img = compute_sigma(reprojected_heights, 2 + 1, depth_to_save, np.array(mask, dtype=np.float32))
            masked_final_sigma = np.array(final_sigma_img, dtype=np.float32) * np.array(mask, dtype=np.float32)

            save_pfm(depth_mask_path, depth_to_save)
            cv2.imwrite(depth_mask_path.replace(".pfm",".png"), visualize_depth(depth_to_save))
            save_pfm(sigma_mask_path, masked_final_sigma)
            cv2.imwrite(sigma_mask_path.replace(".pfm", ".png"),  visualize_depth(masked_final_sigma))
            # print("depth_mask_path")
            print('Iter {}/{},  Res:{}'.format(batch_idx, len(TrainImgLoader), imgs[0].shape))
            # save_pfm(depth_mask_path, height_est_averaged)

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    generate_height_map_masked()
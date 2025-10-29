# Author: Chen Liu (Wuhan University)
# Email: sweet8degree@gmail.com
# Version: V2
# Date: 2025-7-8
# Description: Use Teacher model to infer height map.

import argparse, time, sys, gc
import os.path

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from networkx.algorithms.centrality import group_degree_centrality
from torch.utils.data import DataLoader
from networks.casmvs import CascadeMVSNet
from tools.utils import *
from dataset import find_dataset_def
from dataset.data_io import  save_pfm

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='A PyTorch Implementation')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default="casmvs", help='select model', choices=['samsat', 'red', "casmvs", "ucs", "emvs", "eucs","epnet"])
parser.add_argument('--geo_model', default="rpc", help='select dataset', choices=["rpc", "pinhole"])
parser.add_argument('--dataset_root', default=r'H:\MVS-Dataset\US3D-MVS-Grouped-SINGLE', help='dataset root')

# Resume and save parametersdef organize_single_selected_json(image_folder, out_root, group_name_prefix=""):
#     """
#     Organize image/RPC/height/DSM files into per-combination folders
#     based on each 'selected_*.json' file in the given image folder.
#
#     Args:
#         image_folder (str): Path to the folder containing satellite images and JSON files.
#         out_root (str): Output root directory for grouped combinations.
#         group_name_prefix (str): Optional prefix for group folder names.
#     """
#     block_dir = os.path.dirname(image_folder)
#     folder_base = os.path.basename(block_dir)
#     heightmap2_dir = os.path.join(block_dir, "heightmap2")
#     dsm_dir = os.path.join(block_dir, "dsm")
#
#     for file in os.listdir(image_folder):
#         if file.startswith("selected_") and file.endswith(".json"):
#             json_path = os.path.join(image_folder, file)
#             with open(json_path, "r") as f:
#                 selected = json.load(f).get("selected_images", [])
#
#             if not selected:
#                 print(f"⚠️ Empty selection: {json_path}")
#                 continue
#
#             try:
#                 region, block_id, _ = selected[0].split('_')[0:3]
#             except Exception as e:
#                 print(f"⚠️ Invalid filename format: {selected[0]}")
#                 continue
#
#             combo_idx = file.replace("selected_", "").replace(".json", "")
#             group_name = f"{region}_{block_id}_{combo_idx}"
#             group_dir = os.path.join(out_root, group_name)
#
#             img_dir = os.path.join(group_dir, "image")
#             rpc_dir = os.path.join(group_dir, "rpc")
#             height_dir = os.path.join(group_dir, "height")
#             dsm_out_dir = os.path.join(group_dir, "DSM")
#
#             os.makedirs(img_dir, exist_ok=True)
#             os.makedirs(rpc_dir, exist_ok=True)
#             os.makedirs(height_dir, exist_ok=True)
#             os.makedirs(dsm_out_dir, exist_ok=True)
#
#             for img_file in selected:
#                 # Copy image
#                 src_img = os.path.join(image_folder, img_file)
#                 dst_img = os.path.join(img_dir, img_file)
#                 if not os.path.exists(dst_img) and os.path.exists(src_img):
#                     shutil.copy(src_img, dst_img)
#
#                 # Copy RPC
#                 rpc_file = img_file.replace(".tif", ".rpc")
#                 src_rpc = os.path.join(image_folder, rpc_file)
#                 dst_rpc = os.path.join(rpc_dir, rpc_file)
#                 if os.path.exists(src_rpc) and not os.path.exists(dst_rpc):
#                     shutil.copy(src_rpc, dst_rpc)
#                 elif not os.path.exists(src_rpc):
#                     print(f"⚠️ Missing RPC: {rpc_file}")
#
#                 # Copy height map
#                 height_file = img_file.replace(".tif", "_heightmap.tif")
#                 src_height = os.path.join(heightmap2_dir, height_file)
#                 dst_height = os.path.join(height_dir, height_file)
#                 if os.path.exists(src_height) and not os.path.exists(dst_height):
#                     shutil.copy(src_height, dst_height)
#                 elif not os.path.exists(src_height):
#                     print(f"⚠️ Missing heightmap: {height_file}")
#
#             # Copy DSM file (shared for the whole group)
#             dsm_file_name = f"{region}_{block_id}_DSM.tif"
#             dsm_src = os.path.join(dsm_dir, dsm_file_name)
#             dsm_dst = os.path.join(dsm_out_dir, dsm_file_name)
#             if os.path.exists(dsm_src) and not os.path.exists(dsm_dst):
#                 shutil.copy(dsm_src, dsm_dst)
#             elif not os.path.exists(dsm_src):
#                 print(f"⚠️ Missing DSM file: {dsm_file_name}")
#
#             print(f"✅ Group created: {group_name}")def organize_single_selected_json(image_folder, out_root, group_name_prefix=""):
#     """
#     Organize image/RPC/height/DSM files into per-combination folders
#     based on each 'selected_*.json' file in the given image folder.
#
#     Args:
#         image_folder (str): Path to the folder containing satellite images and JSON files.
#         out_root (str): Output root directory for grouped combinations.
#         group_name_prefix (str): Optional prefix for group folder names.
#     """
#     block_dir = os.path.dirname(image_folder)
#     folder_base = os.path.basename(block_dir)
#     heightmap2_dir = os.path.join(block_dir, "heightmap2")
#     dsm_dir = os.path.join(block_dir, "dsm")
#
#     for file in os.listdir(image_folder):
#         if file.startswith("selected_") and file.endswith(".json"):
#             json_path = os.path.join(image_folder, file)
#             with open(json_path, "r") as f:
#                 selected = json.load(f).get("selected_images", [])
#
#             if not selected:
#                 print(f"⚠️ Empty selection: {json_path}")
#                 continue
#
#             try:
#                 region, block_id, _ = selected[0].split('_')[0:3]
#             except Exception as e:
#                 print(f"⚠️ Invalid filename format: {selected[0]}")
#                 continue
#
#             combo_idx = file.replace("selected_", "").replace(".json", "")
#             group_name = f"{region}_{block_id}_{combo_idx}"
#             group_dir = os.path.join(out_root, group_name)
#
#             img_dir = os.path.join(group_dir, "image")
#             rpc_dir = os.path.join(group_dir, "rpc")
#             height_dir = os.path.join(group_dir, "height")
#             dsm_out_dir = os.path.join(group_dir, "DSM")
#
#             os.makedirs(img_dir, exist_ok=True)
#             os.makedirs(rpc_dir, exist_ok=True)
#             os.makedirs(height_dir, exist_ok=True)
#             os.makedirs(dsm_out_dir, exist_ok=True)
#
#             for img_file in selected:
#                 # Copy image
#                 src_img = os.path.join(image_folder, img_file)
#                 dst_img = os.path.join(img_dir, img_file)
#                 if not os.path.exists(dst_img) and os.path.exists(src_img):
#                     shutil.copy(src_img, dst_img)
#
#                 # Copy RPC
#                 rpc_file = img_file.replace(".tif", ".rpc")
#                 src_rpc = os.path.join(image_folder, rpc_file)
#                 dst_rpc = os.path.join(rpc_dir, rpc_file)
#                 if os.path.exists(src_rpc) and not os.path.exists(dst_rpc):
#                     shutil.copy(src_rpc, dst_rpc)
#                 elif not os.path.exists(src_rpc):
#                     print(f"⚠️ Missing RPC: {rpc_file}")
#
#                 # Copy height map
#                 height_file = img_file.replace(".tif", "_heightmap.tif")
#                 src_height = os.path.join(heightmap2_dir, height_file)
#                 dst_height = os.path.join(height_dir, height_file)
#                 if os.path.exists(src_height) and not os.path.exists(dst_height):
#                     shutil.copy(src_height, dst_height)
#                 elif not os.path.exists(src_height):
#                     print(f"⚠️ Missing heightmap: {height_file}")
#
#             # Copy DSM file (shared for the whole group)
#             dsm_file_name = f"{region}_{block_id}_DSM.tif"
#             dsm_src = os.path.join(dsm_dir, dsm_file_name)
#             dsm_dst = os.path.join(dsm_out_dir, dsm_file_name)
#             if os.path.exists(dsm_src) and not os.path.exists(dsm_dst):
#                 shutil.copy(dsm_src, dsm_dst)
#             elif not os.path.exists(dsm_src):
#                 print(f"⚠️ Missing DSM file: {dsm_file_name}")
#
#             print(f"✅ Group created: {group_name}")
parser.add_argument('--loadckpt', default=r"E:\MVS_Codes\Sat-KD-MVS\US3D_pipeline\checkpoints_US3D\casmvs\rpc\model_000008.ckpt", help='load a specific checkpoint')

# input parameters
parser.add_argument('--view_num', type=int, default=3, help='Number of images.')
# the ref view is set 1. it can set 0, 1 and 2.
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')

# Cascade parameters
parser.add_argument('--ndepths', type=str, default="64,32,8", help='ndepths')
parser.add_argument('--min_interval', type=float, default=0.3, help='min_interval in the bottom stage')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--lamb', type=float, default=1.5, help="lamb in ucs-net")
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])

# run CasMVS model to save depth maps and confidence maps
def infer_depth(view_num):
    # dataset, dataloader
    train_path = args.dataset_root
    MVSDataset = find_dataset_def(args.geo_model, "US3D")
    train_dataset = MVSDataset(train_path, "train", args.view_num, view_num)
    TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=0, drop_last=False,
                                pin_memory=False)

    # model
    model = CascadeMVSNet(min_interval=args.min_interval,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          geo_model=args.geo_model)
    print("===============> Model: Cascade MVS Net ===========>")

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TrainImgLoader):
            sample_cuda = tocuda(sample)
            start_time = time.time()
            outputs = model(sample_cuda["imgs"], sample_cuda["cam_para"], sample_cuda["depth_values"])
            end_time = time.time()
            outputs = tensor2numpy(outputs)
            depth_est = outputs["depth"][0]
            del sample_cuda
            filename = sample["out_name"]
            group_folder = sample["group_folder"][0]
            imgs = sample["imgs"]
            print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(TrainImgLoader), end_time - start_time, imgs[0].shape))
            # save depth maps and confidence maps
            [depth, photometric_confidence, conf_1, conf_2] = outputs["depth"][0], outputs["photometric_confidence"][0], outputs['stage1']["photometric_confidence"][0], outputs['stage2']["photometric_confidence"][0]
            H, W = photometric_confidence.shape
            conf_1 = cv2.resize(conf_1, (W,H))
            conf_2 = cv2.resize(conf_2, (W,H))
            conf_final = photometric_confidence * conf_1 * conf_2

            height_cof_folder = os.path.join(train_path, group_folder,"height_and_cofi")
            if not os.path.exists(height_cof_folder):
                os.makedirs(height_cof_folder)


            depth_filename = os.path.join(train_path, group_folder,"height_and_cofi", str(filename[0]).replace("_RGB","") +'_depth_est.pfm')
            confidence_filename = os.path.join(train_path,group_folder, "height_and_cofi", str(filename[0]).replace("_RGB","")+'_confidence.pfm')
            #save depth maps
            save_pfm(depth_filename, depth_est)
            depth_color = visualize_depth(depth_est)
            cv2.imwrite(depth_filename.replace(".pfm",".png"), depth_color)
            #save confidence maps
            save_pfm(confidence_filename, conf_final)
            cv2.imwrite(confidence_filename.replace(".pfm", ".png"),visualize_depth(conf_final))

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    for i in range(args.view_num):
        infer_depth(i+1)

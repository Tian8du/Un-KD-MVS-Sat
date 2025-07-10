# Author: Chen Liu (Wuhan University)
# Email: sweet8degree@gmail.com
# Version: V2
# Date: 2025-6-29
# Description: Use Teacher model to infer height map.

import argparse, os, time, sys, gc, cv2, signal
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.casmvs import CascadeMVSNet
from tools.utils import *
from dataset import find_dataset_def
from dataset.data_io import load_pfm, save_pfm

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='A PyTorch Implementation')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default="casmvs", help='select model', choices=['samsat', 'red', "casmvs", "ucs", "emvs", "eucs","epnet"])
parser.add_argument('--geo_model', default="rpc", help='select dataset', choices=["rpc", "pinhole"])
parser.add_argument('--use_qc', default=False, help="whether to use Quaternary Cubic Form for RPC warping.")
parser.add_argument('--dataset_root', default=r'E:\Data\WHU_TLC\WHU-TLC', help='dataset root')

# Resume and save parameters
parser.add_argument('--loadckpt', default=r"E:\MVS_Codes\Sat-KD-MVS\checkpoints\casmvs\rpc\model_000008.ckpt", help='load a specific checkpoint')

# input parameters
parser.add_argument('--view_num', type=int, default=3, help='Number of images.')
# the ref view is set 1. it can set 0, 1 and 2.
parser.add_argument('--ref_view', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')

# Cascade parameters
parser.add_argument('--ndepths', type=str, default="64,32,8", help='ndepths')
parser.add_argument('--min_interval', type=float, default=2.5, help='min_interval in the bottom stage')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--lamb', type=float, default=1.5, help="lamb in ucs-net")
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])

# run CasMVS model to save depth maps and confidence maps
def infer_depth():
    # dataset, dataloader
    train_path = "{}/open_dataset_{}/train".format(args.dataset_root, args.geo_model)
    test_path = "{}/open_dataset_{}/test".format(args.dataset_root, args.geo_model)
    MVSDataset = find_dataset_def(args.geo_model, "WHU-TLC")
    train_dataset = MVSDataset(train_path, "train", args.view_num, ref_view=args.ref_view, use_qc=args.use_qc)
    test_dataset = MVSDataset(test_path, "test", args.view_num, ref_view=args.ref_view, use_qc=args.use_qc)
    TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=0, drop_last=False,
                                pin_memory=False)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=0, drop_last=False,
                               pin_memory=False)

    # model
    model = CascadeMVSNet(min_interval=args.min_interval,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          geo_model=args.geo_model, use_qc=args.use_qc)
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
            imgs = sample["imgs"]
            print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(TrainImgLoader), end_time - start_time, imgs[0].shape))

            # save depth maps and confidence maps
            [depth, photometric_confidence, conf_1, conf_2] = outputs["depth"][0], outputs["photometric_confidence"][0], outputs['stage1']["photometric_confidence"][0], outputs['stage2']["photometric_confidence"][0]
            H, W = photometric_confidence.shape
            conf_1 = cv2.resize(conf_1, (W,H))
            conf_2 = cv2.resize(conf_2, (W,H))
            conf_final = photometric_confidence * conf_1 * conf_2

            depth_filename = os.path.join(train_path, "height_and_cofi", str(sample["out_view"][0]), str(filename[0])+'_depth_est.pfm')
            confidence_filename = os.path.join(train_path, "height_and_cofi", str(sample["out_view"][0]), str(filename[0])+'_confidence.pfm')
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
    infer_depth()

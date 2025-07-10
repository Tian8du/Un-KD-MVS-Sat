# Author: Chen Liu (Wuhan University)
# Email: sweet8degree@gmail.com
# Version: V3
# Date: 2025-7-7
# Description: valida the accuracy of MVS

import argparse
import datetime
from tensorboardX import SummaryWriter
import sys
from dataset import find_dataset_def
import torch.backends.cudnn as cudnn
from networks.casmvs import CascadeMVSNet
from networks.ucs import UCSNet
from networks.casred import CascadeREDNet
from torch.utils.data import DataLoader
import torch.nn as nn
from tools.utils import *
torch.autograd.set_detect_anomaly(True)
import time

# if the input size is fixed, the benchmark is true, else false
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation')
parser.add_argument('--mode', default='train', help='train or test', choices=['distillation'])
parser.add_argument('--model', default="casmvs", help='select model', choices=['samsat', 'red', "casmvs", "ucs", "emvs", "eucs","epnet"])
parser.add_argument('--geo_model', default="rpc", help='select dataset', choices=["rpc", "pinhole"])
parser.add_argument('--use_qc', default=False, help="whether to use Quaternary Cubic Form for RPC warping.")
parser.add_argument('--dataset_root', default=r'F:\Data\WHU_TLC\WHU-TLC', help='dataset root')

# Resume and save parameters
parser.add_argument('--loadckpt',default=r"E:\MVS_Codes\Sat-KD-MVS\checkpoints_kd\casmvs\rpc\model_000008.ckpt", help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints_kd', help='the directory to save checkpoints/logs')

# input parameters
parser.add_argument('--view_num', type=int, default=3, help='Number of images.')
# the ref view is set 1. it can set 0, 1 and 2.
parser.add_argument('--ref_view', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')

# Cascade parameters
parser.add_argument('--ndepths', type=str, default="64,32,8", help='ndepths')
parser.add_argument('--min_interval', type=float, default=2.5, help='min_interval in the bottom stage')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--lamb', type=float, default=1.5, help="lamb in ucs-net")

parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')

# training parameters
parser.add_argument('--summary_freq', type=int, default=1, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed')
parser.add_argument('--gpu_id', type=str, default="0")
parser.add_argument('--grad_acc', type=int, default=1, help="the step of grad accumulation")

# parse arguments and check
args = parser.parse_args()

# show the device in use
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Current device: GPU")
else:
    device = torch.device("cpu")
    print("Current device: CPU")

testpath = "{}/open_dataset_{}/test".format(args.dataset_root, args.geo_model)

cur_log_dir = os.path.join(args.logdir, "{}/{}".format(args.model, args.geo_model)).replace("\\", "/")
ck_dir = os.path.join(cur_log_dir, "train").replace("\\", "/")

if not os.path.exists(ck_dir):
    os.makedirs(ck_dir)

# create logger for mode "train" and "testall"
if not os.path.isdir(cur_log_dir):
    os.makedirs(cur_log_dir)

current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
print("current time", current_time_str)

print("creating new summary file")
logger = SummaryWriter(cur_log_dir)

# accumulation grad method
accumulation_steps = args.grad_acc
if accumulation_steps != 1:
    print("Use Grad_accumulation Method")
    args.lr = args.lr * accumulation_steps

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
MVSDataset = find_dataset_def(args.geo_model, "WHU-TLC")
Test_dataset = MVSDataset(testpath, "train", args.view_num, ref_view=args.ref_view, use_qc=args.use_qc)
TestImgLoader = DataLoader(Test_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=False)

# model
model = None
if args.model == "samsat":
    model = ST_SatMVS(min_interval=args.min_interval,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          geo_model=args.geo_model, use_qc=args.use_qc)
elif args.model == "casmvs":
    model = CascadeMVSNet(min_interval=args.min_interval,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          geo_model=args.geo_model, use_qc=args.use_qc)
    print("===============> Model: Cascade MVS Net ===========>")
elif args.model == "ucs":
    model = UCSNet(lamb=args.lamb, stage_configs=[int(nd) for nd in args.ndepths.split(",") if nd],
                   base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                   geo_model=args.geo_model, use_qc=args.use_qc)
    print("===============> Model: UCS-Net ===========>")
elif args.model == "red":
    model = CascadeREDNet(min_interval=args.min_interval,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          geo_model=args.geo_model, use_qc=args.use_qc)
    print("===============> Model: Cascade RED Net ===========>")
else:
    raise Exception("{}? Not implemented yet!".format(args.model))

if torch.cuda.device_count() > 1:
    print("Using multiple GPUs for training")
    model = nn.DataParallel(model)  # Enable multi-GPU parallelism
else:
    print("Using a single GPU or CPU for training")

# move model to cuda.
model = model.cuda() if torch.cuda.is_available() else model
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test():
    """
    Test loop: iterates through the test data loader, runs inference, and prints evaluation metrics for each batch.
    """
    total_time = 0
    for batch_idx, sample in enumerate(TestImgLoader):
        # Get view and name info from sample (can be used for logging)
        bview = sample['out_view'][0]
        bname = sample['out_name'][0]

        start_time = time.time()
        scalar_outputs = test_sample(sample, detailed_summary=True)
        scalar_outputs = {k: float("{0:.6f}".format(v)) for k, v in scalar_outputs.items()}
        batch_time = time.time() - start_time
        total_time += batch_time

        # Print batch-wise evaluation metrics
        print("Iter {}/{}, Name: {}, Time: {:.3f}s | MAE: {:.6f}, RMSE: {:.6f}, Thres1.0m: {:.6f}, Thres2.5m: {:.6f}, Thres7.5m: {:.6f}".format(
            batch_idx+1, len(TestImgLoader), bname, batch_time,
            scalar_outputs["MAE"],
            scalar_outputs["RMSE"],
            scalar_outputs["thres1.0m_error"],
            scalar_outputs["thres2.5m_error"],
            scalar_outputs["thres7.5m_error"]
        ))

        # Release memory for outputs if necessary
        del scalar_outputs
        torch.cuda.empty_cache()

    print("Final total time: {:.3f}s".format(total_time))


@torch.no_grad()
def test_sample(sample, detailed_summary=False):
    """
    Performs forward inference and computes evaluation metrics for a single batch.

    Args:
        sample (dict): Input batch data.
        detailed_summary (bool): If True, additional outputs can be computed (not used here).

    Returns:
        scalar_outputs (dict): Dictionary of computed evaluation metrics.
    """
    model.eval()

    # Move data to CUDA device
    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms     = sample_cuda["mask"]

    # Determine the current stage for multi-stage models
    num_stage = len([nd for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms[f"stage{num_stage}"]
    mask     = mask_ms[f"stage{num_stage}"]

    # Forward pass
    outputs = model(sample_cuda["imgs"],
                    sample_cuda["cam_para"],
                    sample_cuda["depth_values"])

    # Extract predicted depth and photometric confidence according to model type
    if args.model in ("samsat", "ucs"):
        depth_est = outputs["stage3"]["depth_filtered" if args.model == "samsat" else "depth"]
        photometric_conf = outputs["stage3"]["photometric_confidence"]
    else:
        depth_est = outputs["depth"]
        photometric_conf = outputs["stage3"]["photometric_confidence"]

    # Compute evaluation metrics
    mae     = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, 250.).item()
    rmse    = RMSE_metrics(depth_est, depth_gt, mask > 0.5, 250.).item()
    th1     = Thres_metrics(depth_est, depth_gt, mask > 0.5, 1.0).item()
    th2_5   = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2.5).item()
    th7_5   = Thres_metrics(depth_est, depth_gt, mask > 0.5, 7.5).item()

    scalar_outputs = {
        "MAE"            : mae,
        "RMSE"           : rmse,
        "thres1.0m_error": th1,
        "thres2.5m_error": th2_5,
        "thres7.5m_error": th7_5
    }

    return scalar_outputs

if __name__ == '__main__':
    test()

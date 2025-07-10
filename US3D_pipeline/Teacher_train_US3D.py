# Author: Chen Liu (Wuhan University)
# Email: sweet8degree@gmail.com
# Version: V1
# Date: 2025-7-7
# Description: The teacher MVS model using self-training method.

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
from tools.utils import *
from networks.loss import cas_mvsnet_loss
from Loss.unsup_loss import UnSup_Loss
torch.autograd.set_detect_anomaly(True)

# if the input size is fixed, the benchmark is true, else false
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='A PyTorch Implementation')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default="casmvs", help='select model', choices=['samsat', 'red', "casmvs", "ucs", "emvs", "eucs","epnet"])
parser.add_argument('--geo_model', default="rpc", help='select dataset', choices=["rpc", "pinhole"])
parser.add_argument('--use_qc', default=False, help="whether to use Quaternary Cubic Form for RPC warping.")
parser.add_argument('--dataset_root', default=r'H:\MVS-Dataset\Test', help='dataset root')
parser.add_argument('--dataset_name', default=r'US3D', help='dataset name')
parser.add_argument('--place', default='JAX', choices=['JAX', 'OMA', 'JAX+OMA'], help='which place? OMA or JAX?')

# Resume and save parameters
parser.add_argument('--loadckpt', help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints_US3D', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', default=False, help='continue to train the model')

# input parameters
parser.add_argument('--view_num', type=int, default=3, help='Number of images.')
# the ref view is set 1. it can set 0, 1 and 2.
parser.add_argument('--ref_view', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')

# Cascade parameters
parser.add_argument('--ndepths', type=str, default="64,32,8", help='ndepths')
parser.add_argument('--min_interval', type=float, default=0.5, help='min_interval in the bottom stage')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--lamb', type=float, default=1.5, help="lamb in ucs-net")

parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')

# training parameters
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
# Finally, this would not change! 6, 8, 10 and 12.
parser.add_argument('--lrepochs', type=str, default="6,8,10,12:2",
                    help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')

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

# 3. set the train and test path
trainpath = args.dataset_root
testpath = args.dataset_root

if args.resume:
    assert args.mode == "train"
    # assert args.loadckpt is None
if testpath is None:
    testpath = trainpath
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


cur_log_dir = os.path.join(args.logdir, "{}/{}".format(args.model, args.geo_model)).replace("\\", "/")
ck_dir = os.path.join(cur_log_dir, "train").replace("\\", "/")

if not os.path.exists(ck_dir):
    os.makedirs(ck_dir)

# create logger for mode "train" and "testall"
if args.mode == "train":
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

# 8. dataset, dataloader
MVSDataset = find_dataset_def(args.geo_model, args.dataset_name)
train_dataset = MVSDataset(trainpath, "train", args.view_num, ref_view=args.ref_view, use_qc=args.use_qc)
test_dataset = MVSDataset(testpath, "test", args.view_num, ref_view=args.ref_view, use_qc=args.use_qc)

height_range = None
if args.place == "JAX":
    height_range = [-32, 224]
elif args.place == "OMA":
    height_range = [128, 384]
elif args.place == "JAX+OMA":
    height_range = [-32, 384]
# === 新增统一固定height range设置 ===
train_dataset.use_fixed_height_range = True
train_dataset.fixed_height_range = height_range
test_dataset.use_fixed_height_range = True
test_dataset.fixed_height_range = height_range
# =====================================

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

# 9. model
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
elif args.model == "emvs":
    model = CascadeEMVSNet(min_interval=args.min_interval,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          geo_model=args.geo_model, use_qc=args.use_qc)
    print("===============> Model: Cascade EMVS Net ===========>")
elif args.model == "eucs":
    model = eUCSNet(lamb=args.lamb, stage_configs=[int(nd) for nd in args.ndepths.split(",") if nd],
                   base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                   geo_model=args.geo_model, use_qc=args.use_qc)
    print("===============> Model: eUCS-Net ===========>")
elif args.model == "epnet":
    model = EPNet(min_interval=args.min_interval,
                  ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                   depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i]
                  )
    print("===============> Model: Ep-Net ===========>")
else:
    raise Exception("{}? Not implemented yet!".format(args.model))

# move model to cuda.
model = model.cuda() if torch.cuda.is_available() else model

# choose the loss
model_loss = UnSup_Loss()
test_loss = cas_mvsnet_loss

# initial optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)


# load parameters
start_epoch = 1
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(cur_log_dir) if fn.endswith(".ckpt") and len(fn.split("_")) == 2]
    # print(saved_models)
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[1].split(".")[0]))
    # use the latest checkpoint file
    # print(saved_models)
    load_ckpt = os.path.join(cur_log_dir, saved_models[-1])
    print("resuming", load_ckpt)
    state_dict = torch.load(load_ckpt, weights_only=False)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = int(saved_models[-1].split("_")[1].split(".")[0]) + 1
    # print(saved_models)
elif args.loadckpt:
    # load checkpoint file specified by args.load_ckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])

print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


import time, torch
from torch.optim.lr_scheduler import LinearLR, MultiStepLR

def train():
    # ---------- 1.  Parse --lrepochs ------------
    # Example: "10,12,14:2"  →  milestones 10/12/14, decay ×½
    epoch_milestones = [int(e) for e in args.lrepochs.split(':')[0].split(',')]
    lr_gamma         = 1.0 / float(args.lrepochs.split(':')[1])

    # ---------- 2.  LR schedulers ---------------
    warmup_iters = 100                                  # 500 iterations
    sched_warm   = LinearLR(optimizer,
                            start_factor=1/3,
                            total_iters=warmup_iters)    # per-iteration
    sched_main   = MultiStepLR(optimizer,
                               milestones=epoch_milestones,
                               gamma=lr_gamma)           # per-epoch

    global_step = 0

    # ---------- 3.  Training loop ---------------
    for epoch_idx in range(start_epoch, args.epochs + 1):
        print(f"\n==== Epoch {epoch_idx}/{args.epochs} ====")
        model.train()

        for batch_idx, sample in enumerate(TrainImgLoader):
            tic = time.time()

            # ----- forward / backward / update -----
            optimizer.zero_grad(set_to_none=True)
            loss, scalar_out, img_out = train_sample(sample)   # forward pass
            loss.backward()
            optimizer.step()

            # ----- warm-up scheduler (only first 500 iters) -----
            if global_step < warmup_iters:
                sched_warm.step()

            global_step += 1                                   # advance counter

            # ----- logging every args.summary_freq steps -----
            if global_step % args.summary_freq == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                scalars_float = {k: (v if isinstance(v, float) else v.item())
                                 for k, v in scalar_out.items()}
                save_scalars(logger, 'train', scalars_float, global_step)

                print(
                    f"Iter {batch_idx:04}/{len(TrainImgLoader)} | "
                    f"Step {global_step:<7d} | "
                    f"LR {cur_lr:.3e} | "
                    f"Loss {scalars_float['loss']:.3f} | "
                    f"Photo {scalars_float['photometric_loss']:.3f} | "
                    f"Feat {scalars_float['featuremetric_loss']:.3f} | "
                    f"MAE = {scalars_float['MAE']:.3f} | "
                    f"RMSE = {scalars_float['RMSE']:.3f} | "
                    f"PAG1m = {scalars_float['thres1.0m_error']:.3f} | "
                    f"PAG2.5m = {scalars_float['thres2.5m_error']:.3f} | "
                    f"PAG7.5m = {scalars_float['thres7.5m_error']:.3f} | "
                    f"time {time.time() - tic:.2f}s",
                    flush=True
                )
            del scalar_out, img_out  # free VRAM

        # ---------- 4.  Validation loop ----------
        avg_test = DictAverageMeter()
        with torch.no_grad():
            for b_idx, sample in enumerate(TestImgLoader):
                loss_val, sc_out, _ = test_sample(sample)
                avg_test.update(sc_out)
                if b_idx % 20 == 0:
                    print(f"[Val] iter {b_idx:03}/{len(TestImgLoader)} | "
                          f"Loss={sc_out['loss']:.3f} | "
                          f"MAE={sc_out['MAE']:.3f}")

        val_mean = avg_test.mean()
        save_scalars(logger, 'val', val_mean, global_step)
        print("[Val] avg:", {k: f"{v:.3f}" for k, v in val_mean.items()})

        # ---------- 5.  Epoch-level LR decay -----
        sched_main.step()       # called **once per epoch**

        # ---------- 6.  Checkpointing ------------
        if (epoch_idx + 1) % args.save_freq == 0:
            ckpt_path = f"{cur_log_dir}/model_{epoch_idx:06}.ckpt"
            torch.save({'epoch': epoch_idx,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()}, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

        # ---------- 7.  Text log -----------------
        with open(f"{cur_log_dir}/train_record.txt", "a+") as f:
            f.write(f"{epoch_idx} {val_mean}\n")


def test():
    # create output folder
    output_folder = os.path.join(testpath, 'height_result')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    avg_test_scalars = DictAverageMeter()

    total_time = 0
    for batch_idx, sample in enumerate(TestImgLoader):

        bview = sample['out_view'][0]
        bname = sample['out_name'][0]

        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        scalar_outputs = {k: float("{0:.6f}".format(v)) for k, v in scalar_outputs.items()}
        total_time += time.time() - start_time
        print("Iter {}/{}, {}, time = {:3f}, test results = {}".format(batch_idx, len(TestImgLoader),
                                                                       bname, time.time() - start_time, scalar_outputs))

        depth_est = np.float32(np.squeeze(tensor2numpy(image_outputs["depth_est"])))
        prob = np.float32(np.squeeze(tensor2numpy(image_outputs["photometric_confidence"])))

        depth_gt = sample['depth']['stage3']
        mask = sample['mask']['stage3']

        depth_gt = np.float32(np.squeeze(tensor2numpy(depth_gt)))
        mask = (np.squeeze(tensor2numpy(mask))).astype(int)

        depth_gt[mask < 0.5] = -999.0

        del scalar_outputs, image_outputs

    print("final, time = {:3f}, test results = {}".format(total_time, avg_test_scalars.mean()))


def train_sample(sample):
    """
    只做 forward 和指标计算，不做 backward / optimizer.step()！
    返回:
        loss_tensor       - 用于反向传播的 Tensor
        scalar_outputs    - 已转成 python float 的日志字典
        image_outputs     - numpy / cpu 端可视化数据
    """
    model.train()
    sample_cuda = tocuda(sample)

    # -------- forward --------
    outputs = model(sample_cuda["imgs"],
                    sample_cuda["cam_para"],
                    sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    # -------- loss --------
    loss, recon_loss, ssim_loss, \
        smooth_loss, photo_loss, feature_loss = model_loss(
            outputs, sample_cuda["imgs"], sample_cuda["cam_para"],
            dlossw=[float(e) for e in args.dlossw.split(",") if e])

    if torch.isnan(loss):
        raise ValueError("NaN encountered in loss")

    # -------- metrics  --------
    num_stage = len([nd for nd in args.ndepths.split(",") if nd])
    depth_gt = sample_cuda["depth"][f"stage{num_stage}"]
    mask     = sample_cuda["mask"][f"stage{num_stage}"] > 0.5

    scalar_outputs = {
        "loss"              : loss.detach(),
        "photometric_loss"  : photo_loss.item(),
        "featuremetric_loss": feature_loss.item(),
        "MAE"               : AbsDepthError_metrics(depth_est, depth_gt, mask, 250.).item(),
        "RMSE"              : RMSE_metrics(depth_est, depth_gt, mask, 250.0).item(),
        "thres1.0m_error"   : Thres_metrics(depth_est, depth_gt, mask, 1.0).item(),
        "thres2.5m_error"   : Thres_metrics(depth_est, depth_gt, mask, 2.5).item(),
        "thres7.5m_error"   : Thres_metrics(depth_est, depth_gt, mask, 7.5).item(),
    }

    image_outputs = {
        "depth_est": depth_est.detach().cpu(),
        "ref_img"  : sample["imgs"][:, 0]
    }

    return loss, scalar_outputs, image_outputs


def train_sample2(sample, detailed_summary=False):
    model.train()
    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]
    outputs = model(sample_cuda["imgs"], sample_cuda["cam_para"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    loss, depth_loss = test_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e], depth_values=sample_cuda["depth_values"])

    scalar_outputs = {"loss": loss, "depth_loss": depth_loss}
    image_outputs = {"depth_est": depth_est, "depth_gt": depth_gt,
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"]}

    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, 250.0)
        scalar_outputs["RMSE"] = RMSE_metrics(depth_est, depth_gt, mask > 0.5, 250.0)
        scalar_outputs["thres1.0m_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 1.0)
        scalar_outputs["thres2.5m_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2.5)
        scalar_outputs["thres7.5m_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 7.5)

    return loss, scalar_outputs, image_outputs

@torch.no_grad()
def test_sample(sample, detailed_summary=False):
    """
    评估阶段：前向推理 + 统计指标
    返回:
        loss_val        - float
        scalar_outputs  - dict{str: float}
        image_outputs   - dict{str: Tensor/ndarray}
    """
    model.eval()

    # -------- 数据搬到 CUDA --------
    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms     = sample_cuda["mask"]

    num_stage = len([nd for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms[f"stage{num_stage}"]
    mask     = mask_ms[f"stage{num_stage}"]

    # -------- 前向 --------
    outputs = model(sample_cuda["imgs"],
                    sample_cuda["cam_para"],
                    sample_cuda["depth_values"])

    # 根据模型名字取相应 stage 的 depth
    if args.model in ("samsat", "ucs"):
        depth_est = outputs["stage3"]["depth_filtered" if args.model == "samsat" else "depth"]
        photometric_conf = outputs["stage3"]["photometric_confidence"]
    else:
        depth_est = outputs["depth"]
        photometric_conf = outputs["stage3"]["photometric_confidence"]

    # -------- 计算 loss --------
    loss_tensor, depth_loss = test_loss(
        outputs, depth_gt_ms, mask_ms,
        dlossw=[float(e) for e in args.dlossw.split(",") if e],
        depth_values=sample_cuda["depth_values"]
    )
    loss_val = loss_tensor.item()

    # -------- 指标 --------
    mae     = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, 250.).item()
    rmse    = RMSE_metrics(depth_est, depth_gt, mask > 0.5, 250.).item()
    th1     = Thres_metrics(depth_est, depth_gt, mask > 0.5, 1.0).item()
    th2_5   = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2.5).item()
    th7_5   = Thres_metrics(depth_est, depth_gt, mask > 0.5, 7.5).item()

    scalar_outputs = {
        "loss"           : loss_val,
        "depth_loss"     : depth_loss.item(),
        "MAE"            : mae,
        "RMSE"           : rmse,
        "thres1.0m_error": th1,
        "thres2.5m_error": th2_5,
        "thres7.5m_error": th7_5
    }

    # -------- 可视化输出 --------
    image_outputs = {
        "depth_est"            : depth_est.detach().cpu(),
        "photometric_confidence": photometric_conf.detach().cpu(),
        "depth_gt"             : sample["depth"]["stage1"],
        "ref_img"              : sample["imgs"][:, 0],
        "mask"                 : sample["mask"]["stage1"]
    }
    if detailed_summary:
        err_map = (depth_est - depth_gt).abs() * mask
        image_outputs["errormap"] = err_map.detach().cpu()

    return loss_val, scalar_outputs, image_outputs

if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()


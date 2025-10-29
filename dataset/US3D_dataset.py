from torch.utils.data import Dataset
from dataset.data_io import *
from dataset.preprocess import *
from dataset.gen_list import *
import rasterio

# This is the dataset for US3D-MVS, DeepLearning.
class US3DDataset(Dataset):
    """
    """
    def __init__(self, data_folder, mode, view_num, ref_view):
        """
        the Initial
        Args:
            data_folder: the folder path for this dataset
            mode:  the mode for training
            view_num:  the number of multiple satellite images
            use_qc:  whether or not using QC
        """
        super(US3DDataset, self).__init__()
        self.data_folder = data_folder
        self.mode = mode
        self.view_num = view_num
        self.ref_view = ref_view

        self.use_fixed_height_range = False
        self.fixed_height_range = None

        assert self.mode in ["train", "val", "test", "pred", "distillation"]
        self.sample_list = self.build_list()
        self.sample_num = len(self.sample_list)

    def build_list(self):
        sample_list = gen_imgs(self.data_folder,self.view_num)
        if self.mode == "pred":
            sample_list = sample_list
        elif self.mode == "train":
            sample_list = sample_list
        elif self.mode == "test":
            total = len(sample_list)
            keep = int(total * 0.1)
            sample_list = random.sample(sample_list, keep)
        elif self.mode == "val":
            total = len(sample_list)
            keep = int(total * 0.1)
            sample_list = sample_list[:keep]

        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def get_sample(self, idx):
        samples = self.sample_list[idx]
        imgs = samples["image"]
        rpcs = samples["rpc"]
        heights = samples["height"]
        imgs, rpcs, heights = swap_ref_view(imgs, rpcs, heights, self.ref_view)

        centered_images = []
        rpc_paramters = []

        # Height
        with rasterio.open(heights[0]) as src:
            depth_image = src.read(1).astype(np.float32)

        # ================== 新增 depth_values 决策逻辑 ==================
        # 判断是否使用固定height range
        if hasattr(self, "use_fixed_height_range") and self.use_fixed_height_range and hasattr(self,
                                                                                               "fixed_height_range"):
            depth_min, depth_max = self.fixed_height_range
            # print(f"[Info] Using fixed height range: {depth_min} ~ {depth_max}")
        else:
            _, depth_max, depth_min = load_rpc_as_array(rpcs[0])
            # print(f"[Info] Using RPC-derived height range: {depth_min} ~ {depth_max}")

        depth_values = np.array([depth_min, depth_max], dtype=np.float32)
        # ================================================================

        for view in range(self.view_num):
            if self.mode == "train":
                image = image_augment(read_img(imgs[view]))
            else:
                image = read_img(imgs[view])
            image = np.asarray(image)

            # RPC
            rpc, _, _ = load_rpc_as_array(rpcs[view])
            rpc_paramters.append(rpc)
            centered_images.append(center_image(image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        rpc_paramters = np.array(rpc_paramters)

        # Mask
        mask = np.float32((depth_image >= depth_min) * 1.0) * np.float32((depth_image <= depth_max) * 1.0)

        h, w = depth_image.shape
        depth_ms = {
            "stage1": cv2.resize(depth_image, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_image, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_image
        }
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask
        }

        # Multi-stage RPC parameters
        stage2_rpc = rpc_paramters.copy()
        stage2_rpc[:, 0] /= 2
        stage2_rpc[:, 1] /= 2
        stage2_rpc[:, 5] /= 2
        stage2_rpc[:, 6] /= 2

        stage3_rpc = rpc_paramters.copy()
        stage3_rpc[:, 0] /= 4
        stage3_rpc[:, 1] /= 4
        stage3_rpc[:, 5] /= 4
        stage3_rpc[:, 6] /= 4

        rpc_paramters_ms = {
            "stage1": stage3_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_paramters
        }

        path = imgs[0]
        group_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
        out_name = os.path.splitext(os.path.basename(path))[0]

        return {"imgs": centered_images,
                "cam_para": rpc_paramters_ms,
                "depth": depth_ms,
                "mask": mask_ms,
                "depth_values": depth_values,
                "out_name": out_name,
                "group_folder": group_folder
                }

    def get_pred_sample(self, idx):
        samples = self.sample_list[idx]
        imgs = samples["image"]
        rpcs = samples["rpc"]
        heights = samples["height"]
        dsm = samples["DSM"]

        centered_images = []
        rpc_paramters = []

        # Height
        with rasterio.open(heights[0]) as src:
            depth_image = src.read(1).astype(np.float32)

        # ================== 新增 depth_values 决策逻辑 ==================
        # 判断是否使用固定height range
        if hasattr(self, "use_fixed_height_range") and self.use_fixed_height_range and hasattr(self,
                                                                                               "fixed_height_range"):
            depth_min, depth_max = self.fixed_height_range
            # print(f"[Info] Using fixed height range: {depth_min} ~ {depth_max}")
        else:
            _, depth_max, depth_min = load_rpc_as_array(rpcs[0])
            # print(f"[Info] Using RPC-derived height range: {depth_min} ~ {depth_max}")

        depth_values = np.array([depth_min, depth_max], dtype=np.float32)
        # ================================================================

        for view in range(self.view_num):
            if self.mode == "train":
                image = image_augment(read_img(imgs[view]))
            else:
                image = read_img(imgs[view])
            image = np.asarray(image)

            # RPC
            rpc, _, _ = load_rpc_as_array(rpcs[view])
            rpc_paramters.append(rpc)
            centered_images.append(center_image(image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        rpc_paramters = np.array(rpc_paramters)

        # Mask
        mask = np.float32((depth_image >= depth_min) * 1.0) * np.float32((depth_image <= depth_max) * 1.0)

        h, w = depth_image.shape
        depth_ms = {
            "stage1": cv2.resize(depth_image, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_image, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_image
        }
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask
        }

        # Multi-stage RPC parameters
        stage2_rpc = rpc_paramters.copy()
        stage2_rpc[:, 0] /= 2
        stage2_rpc[:, 1] /= 2
        stage2_rpc[:, 5] /= 2
        stage2_rpc[:, 6] /= 2

        stage3_rpc = rpc_paramters.copy()
        stage3_rpc[:, 0] /= 4
        stage3_rpc[:, 1] /= 4
        stage3_rpc[:, 5] /= 4
        stage3_rpc[:, 6] /= 4

        rpc_paramters_ms = {
            "stage1": stage3_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_paramters
        }

        path = imgs[0]
        out_name = os.path.splitext(os.path.basename(path))[0]

        return {"imgs": centered_images,
                "cam_para": rpc_paramters_ms,
                "depth": depth_ms,
                "mask": mask_ms,
                "depth_values": depth_values,
                "out_name": out_name,
                "dsm_path": dsm
                }

    def get_dsm_sample(self, idx):
        # 生成点云提取DSM需要 （1） 每张影像的深度图，每张影像的rpc，这个区域对应的DSM（范围，投影）
        samples = self.sample_list[idx]
        imgs = samples["image"]
        rpcs = samples["rpc"]
        heights = samples["height"]
        dsm = samples["DSM"]

        centered_images = []
        rpc_paramters = []

        # Height
        with rasterio.open(heights[0]) as src:
            depth_image = src.read(1).astype(np.float32)

        # ================== 新增 depth_values 决策逻辑 ==================
        # 判断是否使用固定height range
        if hasattr(self, "use_fixed_height_range") and self.use_fixed_height_range and hasattr(self,
                                                                                               "fixed_height_range"):
            depth_min, depth_max = self.fixed_height_range
            # print(f"[Info] Using fixed height range: {depth_min} ~ {depth_max}")
        else:
            _, depth_max, depth_min = load_rpc_as_array(rpcs[0])
            # print(f"[Info] Using RPC-derived height range: {depth_min} ~ {depth_max}")

        depth_values = np.array([depth_min, depth_max], dtype=np.float32)
        # ================================================================

        for view in range(self.view_num):
            if self.mode == "train":
                image = image_augment(read_img(imgs[view]))
            else:
                image = read_img(imgs[view])
            image = np.asarray(image)

            # RPC
            rpc, _, _ = load_rpc_as_array(rpcs[view])
            rpc_paramters.append(rpc)
            centered_images.append(center_image(image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        rpc_paramters = np.array(rpc_paramters)

        # Mask
        mask = np.float32((depth_image >= depth_min) * 1.0) * np.float32((depth_image <= depth_max) * 1.0)

        h, w = depth_image.shape
        depth_ms = {
            "stage1": cv2.resize(depth_image, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_image, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_image
        }
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask
        }

        # Multi-stage RPC parameters
        stage2_rpc = rpc_paramters.copy()
        stage2_rpc[:, 0] /= 2
        stage2_rpc[:, 1] /= 2
        stage2_rpc[:, 5] /= 2
        stage2_rpc[:, 6] /= 2

        stage3_rpc = rpc_paramters.copy()
        stage3_rpc[:, 0] /= 4
        stage3_rpc[:, 1] /= 4
        stage3_rpc[:, 5] /= 4
        stage3_rpc[:, 6] /= 4

        rpc_paramters_ms = {
            "stage1": stage3_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_paramters
        }

        path = imgs[0]
        out_name = os.path.splitext(os.path.basename(path))[0]

        return {"imgs": centered_images,
                "cam_para": rpc_paramters_ms,
                "depth": depth_ms,
                "mask": mask_ms,
                "depth_values": depth_values,
                "out_name": out_name,
                "dsm_path": dsm
                }


    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        if self.mode != "pred":
            return self.get_sample(idx)
        else:
            return self.get_pred_sample(idx)

def gen_imgs(root_dir, view_num=None):
    """
    Scan all grouped folders and return the image/rpc/height/DSM path list for each group.

    Args:
        root_dir (str): Root directory containing all groups (e.g., US3D-MVS-Grouped-JAX).
        view_num (int, optional): If specified, limits the number of views per group.

    Returns:
        List[dict]: Each dict contains 'group', 'image', 'rpc', 'height', and 'dsm' keys.
    """
    results = []
    for entry in sorted(os.listdir(root_dir)):
        group_dir = os.path.join(root_dir, entry)
        if not os.path.isdir(group_dir):
            continue

        image_dir = os.path.join(group_dir, "image")
        rpc_dir = os.path.join(group_dir, "rpc")
        height_dir = os.path.join(group_dir, "height")
        dsm_dir = os.path.join(group_dir, "DSM")

        if not all(os.path.isdir(sub) for sub in [image_dir, rpc_dir, height_dir, dsm_dir]):
            continue

        img_list = sorted([os.path.join(image_dir, f)for f in os.listdir(image_dir)if f.lower().endswith(".tif")])
        rpc_list = sorted([os.path.join(rpc_dir, f)for f in os.listdir(rpc_dir)if f.lower().endswith(".rpc")])
        height_list = sorted([os.path.join(height_dir, f)for f in os.listdir(height_dir)if f.lower().endswith(".tif")])

        # Get the single DSM .tif file
        dsm_files = [f for f in os.listdir(dsm_dir) if f.lower().endswith(".tif")]
        if len(dsm_files) != 1:
            print(f"⚠️ Invalid DSM file count in group: {entry}")
            continue
        dsm_path = os.path.join(dsm_dir, dsm_files[0])

        # Limit number of views if requested
        if view_num is not None:
            img_list = img_list[:view_num]
            rpc_list = rpc_list[:view_num]
            height_list = height_list[:view_num]

        results.append({
            "group": entry,
            "image": img_list,
            "rpc": rpc_list,
            "height": height_list,
            "dsm": dsm_path
        })

    return results


def load_rpc_from_image(image_path):
    ds = gdal.Open(image_path)
    if ds is None:
        raise Exception(f"无法打开影像文件: {image_path}")

    metadata = ds.GetMetadata()
    rpc_info = {}
    for key, value in metadata.items():
        if key.startswith("RPC_"):
            rpc_info[key] = value

    if len(rpc_info) == 0:
        raise Exception("没有找到RPC信息!")
    for key, value in rpc_info.items():
        print(f"{key}: {value}")

    return rpc_info

def swap_ref_view(imgs, rpcs, heights, ref_view):
    # ref_view是1-based，转成0-based
    idx = ref_view - 1
    if idx == 0:
        return imgs, rpcs, heights  # 已经在第一个，不需要交换
    # 交换第idx个和第0个
    imgs_new = imgs.copy()
    rpcs_new = rpcs.copy()
    heights_new = heights.copy()
    imgs_new[0], imgs_new[idx] = imgs_new[idx], imgs_new[0]
    rpcs_new[0], rpcs_new[idx] = rpcs_new[idx], rpcs_new[0]
    heights_new[0], heights_new[idx] = heights_new[idx], heights_new[0]
    return imgs_new, rpcs_new, heights_new


if __name__ == "__main__":
    # gen_imgs(r"E:\Data\US3D\US3D-MVS\JAX",3,1)
    dataset = US3DDataset(r"H:\MVS-Dataset\Test\JAX","test",5,1)
    print("OK")
    print(dataset)
    print("OK")
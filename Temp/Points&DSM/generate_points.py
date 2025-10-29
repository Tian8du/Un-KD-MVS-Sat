import numpy as np
from typing import List, Tuple, Optional
from tools.rpc_core import RPCModelParameter, load_rpc_as_array
from tools.rpc_filter import filter_depth  # 你的深度过滤函数
from tools.io import load_pfm, write_las
from pyproj import Transformer, CRS
import os


def get_img_size(pfm_path):
    data = load_pfm(pfm_path)
    return data.shape[::-1]  # (W, H)

def get_utm_proj_from_rpc(rpc_array):
    rpc = RPCModelParameter(rpc_array)
    lon, lat = rpc.LONG_OFF, rpc.LAT_OFF
    utm_zone = int((lon + 180) / 6) + 1
    north = lat >= 0
    epsg_code = 32600 + utm_zone if north else 32700 + utm_zone
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)


def generate_point_clouds(height_map_paths: List[str],
                          rpc_paths: List[str],
                          las_path,
                          img_size: Tuple[int, int],
                          p_thred: float,
                          d_thred: float,
                          geo_consist_num: int,
                          proj,
                          ref_view: int = 0,
                          confidence_ratio: float = 0.2) -> Optional[np.ndarray]:
    """
    Generate 3D point clouds from depth maps and RPCs.

    Args:
        height_map_paths: List of paths to .pfm depth maps.
        rpc_paths: List of paths to .rpc files (same order).
        img_size: (W, H) of the image (should match depth maps).
        p_thred: probability ratio threshold for consistency.
        d_thred: depth deviation threshold for consistency.
        geo_consist_num: number of views required to agree.
        proj: pyproj Transformer (e.g., WGS84 -> UTM).
        ref_view: index of reference view.
        confidence_ratio: filter confidence.

    Returns:
        points: (N, 3) numpy array of [X, Y, Z] in projected coordinates.
    """
    assert len(height_map_paths) == len(rpc_paths), "Input path list length mismatch"

    height_maps = [load_pfm(p) for p in height_map_paths]
    rpcs = [load_rpc_as_array(p)[0] for p in rpc_paths]

    heights = np.stack(height_maps, axis=0)
    rpc_stack = np.stack(rpcs, axis=0)

    # Step 1: geometric consistency filtering
    mask, height_avg = filter_depth(
        heights, rpc_stack,
        p_ratio=p_thred,
        d_ratio=d_thred,
        geo_consist_num=geo_consist_num,
        prob=None,
        confidence_ratio=confidence_ratio
    )

    H, W = img_size[1], img_size[0]
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    height_flat = height_avg.reshape(-1)
    mask_flat = mask.reshape(-1)

    # Step 2: extract valid pixels
    x_valid = xx[mask_flat].astype(np.float64)
    y_valid = yy[mask_flat].astype(np.float64)
    h_valid = height_flat[mask_flat].astype(np.float64)

    if len(x_valid) == 0:
        return None

    # Step 3: use RPC to map image to geographic coordinates
    ref_rpc = RPCModelParameter(rpc_stack[ref_view])
    lat, lon = ref_rpc.RPC_PHOTO2OBJ(x_valid, y_valid, h_valid)
    geopts = np.stack((lon, lat), axis=-1)

    # Step 4: project lat/lon to planar coordinates
    x_proj, y_proj = proj.transform(geopts[:, 1], geopts[:, 0])
    points = np.stack((x_proj, y_proj, h_valid), axis=-1)

    # Step 5: save to LAS file
    write_las(las_path, points)
    return points


if __name__ == "__main__":
    base_dir = r"/Temp/Test/JAX_004_0"
    views = ["013", "014", "016"]

    height_paths = [
        os.path.join(base_dir, "height_and_cofi", f"JAX_004_{vid}_depth_est.pfm")
        for vid in views
    ]
    rpc_paths = [
        os.path.join(base_dir, "rpc", f"JAX_004_{vid}_RGB.rpc")
        for vid in views
    ]
    out_las = os.path.join(base_dir, "JAX_004_0_points.las")

    # Auto read image size
    img_size = get_img_size(height_paths[0])  # (W, H)

    # Load RPC and auto get projection
    rpc_array = load_rpc_as_array(rpc_paths[0])[0]
    proj = get_utm_proj_from_rpc(rpc_array)

    # Generate point cloud
    points = generate_point_clouds(
        height_map_paths=height_paths,
        rpc_paths=rpc_paths,
        las_path=out_las,
        img_size=img_size,
        p_thred=2,
        d_thred=500,
        geo_consist_num=2,
        proj=proj,
        ref_view=0,
        confidence_ratio=0.2
    )

    if points is not None:
        print(f"✅ Point cloud saved: {out_las}, total points: {points.shape[0]}")
    else:
        print("❌ No valid points generated.")
    pass
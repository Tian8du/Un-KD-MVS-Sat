
import numpy as np
import cv2
import numpy_groupies as npg


def proj_to_grid(points, xoff, yoff, xresolution, yresolution, xsize, ysize):
    row = np.floor((yoff - points[:, 1]) / xresolution).astype(dtype=int)
    col = np.floor((points[:, 0] - xoff) / yresolution).astype(dtype=int)

    points_group_idx = row * xsize + col
    points_val = points[:, 2]

    # remove points that lie out of the dsm boundary
    mask = ((row >= 0) * (col >= 0) * (row < ysize) * (col < xsize)) > 0

    # print("mask num:", np.sum(mask.astype(np.int)))

    points_group_idx = points_group_idx[mask]
    points_val = points_val[mask]

    # create a place holder for all pixels in the dsm
    group_idx = np.arange(xsize * ysize).astype(dtype=int)
    group_val = np.empty(xsize * ysize)
    group_val.fill(np.nan)

    # concatenate place holders with the real valuies, then aggregate
    group_idx = np.concatenate((group_idx, points_group_idx))
    group_val = np.concatenate((group_val, points_val))

    dsm = npg.aggregate(group_idx, group_val, func='nanmax', fill_value=np.nan)
    dsm = dsm.reshape((ysize, xsize))

    # try to fill very small holes
    dsm_new = dsm.copy()
    nan_places = np.argwhere(np.isnan(dsm_new))
    for i in range(nan_places.shape[0]):
        row = nan_places[i, 0]
        col = nan_places[i, 1]
        neighbors = []
        for j in range(row-1, row+2):
            for k in range(col-1, col+2):
                if j >= 0 and j < dsm_new.shape[0] and k >=0 and k < dsm_new.shape[1]:
                    val = dsm_new[j, k]
                    if not np.isnan(val):
                        neighbors.append(val)

        if neighbors:
            dsm[row, col] = np.median(neighbors)

    return dsm


def produce_dsm_from_points(points, ul_e, ul_n, xunit, yunit, e_size, n_size):
    # write dsm to tif
    dsm = proj_to_grid(points, ul_e, ul_n, xunit, yunit, e_size, n_size)
    # median filter
    # dsm = np.zeros((n_size, e_size))
    dsm = cv2.medianBlur(dsm.astype(np.float32), 3)

    return dsm


def fuse_dsm(all_dsm):
    cnt = len(all_dsm)
    if cnt == 1:
        return all_dsm[0]

    all_dsm = np.stack(all_dsm, axis=-1)
    if cnt == 2:
        fused_dsm = np.mean(all_dsm, axis=-1)
        return fused_dsm

    # reject two measurements
    num_measurements = cnt - np.sum(np.isnan(all_dsm), axis=2, keepdims=True)
    mask = np.tile(num_measurements <= 2, (1, 1, cnt))
    all_dsm[mask] = np.nan

    # reject outliers based on MAD statistics
    all_dsm_median = np.nanmedian(all_dsm, axis=2, keepdims=True)
    all_dsm_mad = np.nanmedian(np.abs(all_dsm - all_dsm_median), axis=2, keepdims=True)
    outlier_mask = np.abs(all_dsm - all_dsm_median) > all_dsm_mad
    all_dsm[outlier_mask] = np.nan
    all_dsm_mean_no_outliers = np.nanmean(all_dsm, axis=2)

    # median filter
    all_dsm_mean_no_outliers = cv2.medianBlur(all_dsm_mean_no_outliers.astype(np.float32), 3)

    return all_dsm_mean_no_outliers


def generate_dsm_from_points(
        points: np.ndarray,
        ul_e: float,
        ul_n: float,
        xunit: float,
        yunit: float,
        xsize: int,
        ysize: int,
        out_dsm_path: str,
        proj_wkt: str,
        nodata_val: float = -9999.0,
        apply_fusion: bool = False
    ):
    """
    从点云生成 DSM，并写入 GeoTIFF 文件。

    Args:
        points: (N, 3) ndarray, [X, Y, Z].
        ul_e, ul_n: DSM 左上角投影坐标.
        xunit, yunit: DSM 分辨率（米）.
        xsize, ysize: DSM 图像大小.
        out_dsm_path: 输出 TIF 文件路径.
        proj_wkt: 投影信息.
        nodata_val: 无效值.
        apply_fusion: 是否使用融合滤波（默认 False).
    """
    dsm = produce_dsm_from_points(points, ul_e, ul_n, xunit, yunit, xsize, ysize)

    if apply_fusion:
        dsm = fuse_dsm([dsm])

    dsm[np.isnan(dsm)] = nodata_val

    geotransform = [ul_e - xunit / 2, xunit, 0, ul_n - (-yunit) / 2, 0, -yunit]

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(out_dsm_path, xsize, ysize, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(proj_wkt)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata_val)
    band.WriteArray(dsm)
    ds.FlushCache()
    ds = None

import laspy
import open3d as o3d
import numpy as np

# 读取 .las 文件
las = laspy.read(r"E:\MVS_Codes\Sat-KD-MVS\Test\JAX_004_0\JAX_004_0_points.las")
points = np.vstack((las.x, las.y, las.z)).transpose()

# 转为 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 可视化
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Viewer")

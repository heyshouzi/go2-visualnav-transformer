import open3d as o3d
import numpy as np

def visualize_ply_file(ply_file_path: str, target_num_points: int):
    """
    Visualize a PLY file using Open3D, display the XYZ coordinate frame,
    remove statistical outliers, and downsample or pad the point cloud to a specified number of points.

    Parameters:
    ply_file_path (str): Path to the PLY file to be visualized.
    target_num_points (int): Target number of points after downsampling or padding.
    """
    # Load the PLY file
    pcd = o3d.io.read_point_cloud(ply_file_path)
    
    # Check if the point cloud was loaded successfully
    if not pcd.has_points():
        print("Error: No points found in the PLY file.")
        return
    
    # Print the number of points before outlier removal
    print(f"Number of points before outlier removal: {len(pcd.points)}")
    
    # Apply statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_filtered = pcd.select_by_index(ind)
    
    # Print the number of points after outlier removal
    print(f"Number of points after outlier removal: {len(pcd_filtered.points)}")
    
    # Get the number of points after outlier removal
    current_points = len(pcd_filtered.points)
    
    if current_points < target_num_points:
        # 计算需要补充的点数
        missing_points = target_num_points - current_points

        # 将现有的点云数据转为numpy数组
        points = np.asarray(pcd_filtered.points)

        # 创建一个全为零的点云数据，用于补充
        zero_points = np.zeros((missing_points, 3))

        # 将原始点云和零点数据拼接
        new_points = np.vstack((points, zero_points))

        # 创建新的点云对象并更新点
        pcd_downsampled = o3d.geometry.PointCloud()
        pcd_downsampled.points = o3d.utility.Vector3dVector(new_points)
    else:
         # 使用随机下采样
        sampling_ratio = target_num_points / current_points
        pcd_downsampled = pcd_filtered.random_down_sample(sampling_ratio=sampling_ratio)
    
    # Print the number of points after downsampling or padding
    print(f"Number of points after downsampling or padding: {len(pcd_downsampled.points)}")
    
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY File Visualization", width=800, height=600)
    
    # Add the downsampled or padded point cloud to the visualizer
    vis.add_geometry(pcd_downsampled)
    
    # Create a coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    # Add the coordinate frame to the visualizer
    vis.add_geometry(coordinate_frame)
    
    # Set the background color to white
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    
    # Run the visualizer
    vis.run()
    
    # Destroy the window after closing
    vis.destroy_window()

if __name__ == "__main__":
    # Example usage
    ply_file_path = "/home/zhineng/wl/go2-visualnav-transformer/datasets/meadow/meadow_nomad_xunshan_3_2/26.ply"
    target_num_points = 4000 # Specify the target number of points
    visualize_ply_file(ply_file_path, target_num_points)
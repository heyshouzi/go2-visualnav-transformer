import open3d as o3d
import numpy as np

def visualize_ply_file(ply_file_path: str):
    """
    Visualize a PLY file using Open3D, display the XYZ coordinate frame,
    and remove statistical outliers.

    Parameters:
    ply_file_path (str): Path to the PLY file to be visualized.
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
    
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY File Visualization", width=800, height=600)
    
    # Add the filtered point cloud to the visualizer
    vis.add_geometry(pcd_filtered)
    
    # Create a coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coorgitdinate_frame(size=1.0, origin=[0, 0, 0])
    
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
    ply_file_path = "/home/zhineng/wl/go2-visualnav-transformer/datasets/meadow/meadow_nomad_xunshan_4_10/1.ply"
    visualize_ply_file(ply_file_path)
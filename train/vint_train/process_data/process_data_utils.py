import numpy as np
import io
import os
import rosbag
from PIL import Image
import cv2
from typing import Any, Tuple, List, Dict
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import torchvision.transforms.functional as TF

IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = 4 / 3


def save_point_cloud_to_ply(point_cloud, output_path):
    """
    将点云保存为PLY格式
    :param point_cloud: 一个(N, 3)的numpy数组 包含点云坐标
    :param output_path: 输出路径 保存为PLY文件
    """
    # 创建一个open3d点云对象
    pcd = o3d.geometry.PointCloud()
    # 设置点云坐标
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    # 保存PLY文件
    o3d.io.write_point_cloud(output_path, pcd)
# utils



def process_images(im_list: List, img_process_func) -> List:
    """
    Process image data from a topic that publishes ros images into a list of PIL images
    """
    images = []
    for img_msg in im_list:
        img = img_process_func(img_msg)
        images.append(img)
    return images


def process_lidar(lidar_list: List, lidar_process_func) -> List:
    """
    Process LiDAR data from a topic that publishes sensor_msgs/PointCloud2 to an Open3D PointCloud object.
    This function will convert the list of PointCloud2 message to the list of np.ndarray and return it.

    Args:
        lidar_list (List[sensor_msgs/PointCloud2]): The list of ROS PointCloud2 message
        lidar_process_func (Callable): Function to process each PointCloud2 message. Defaults to filter_lidar.

    Returns:
        List[np.ndarray]: The processed point cloud data as a list of numpy arrays
    """
    # Convert PointCloud2 message to numpy array
    processed_lidar_list = []
    for lidar_msg in lidar_list:
        lidar = lidar_process_func(lidar_msg)
        processed_lidar_list.append(lidar)

    return processed_lidar_list


def filter_lidar(msg) -> np.ndarray:
    # TODO：将点云数据变换到相机所在的坐标系
    """
    过滤并提取 sensor_msgs/PointCloud2 消息中的 XYZ 点云数据，返回 numpy 数组格式 (N, 3)

    Args:
        msg (sensor_msgs.PointCloud2): ROS PointCloud2 消息

    Returns:
        np.ndarray: 包含点云 XYZ 坐标的 numpy 数组，形状为 (N, 3)
    """
    # 提取点云数据
    point_cloud_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    
    # 将点云数据转换为 numpy 数组
    point_cloud_array = np.array(point_cloud_list, dtype=np.float32)
    
    # 过滤掉 z 坐标过高的点（例如 z_max = 3.5）
    z_max = 3.5
    filtered_point_cloud_array = point_cloud_array[point_cloud_array[:, 2] <= z_max]
    
    # 过滤掉地面（例如 z_ground = 0.1）
    z_ground = 0.1
    filtered_point_cloud_array = filtered_point_cloud_array[filtered_point_cloud_array[:, 2] >= z_ground]
    
    # 将 numpy 数组转换为 Open3D PointCloud 对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_point_cloud_array)
    
    # 统计去噪：去除异常点
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_filtered = pcd.select_by_index(ind)
    
    # 获取过滤后的点云数据
    filtered_point_cloud_array = np.asarray(pcd_filtered.points)
    
    # 目标点云数量
    target_num_points = 4000
    
    # 获取当前点云数量
    current_points = len(filtered_point_cloud_array)
    
    if current_points > target_num_points:
        # 随机下采样
        sampling_ratio = target_num_points / current_points
        pcd_downsampled = pcd_filtered.random_down_sample(sampling_ratio=sampling_ratio)
        final_point_cloud_array = np.asarray(pcd_downsampled.points)
    elif current_points < target_num_points:
        # 用零填充
        missing_points = target_num_points - current_points
        zero_points = np.zeros((missing_points, 3), dtype=np.float32)
        final_point_cloud_array = np.vstack((filtered_point_cloud_array, zero_points))
    else:
        final_point_cloud_array = filtered_point_cloud_array
    # 确保最终点云数组的形状为 (4000, 3)
    if final_point_cloud_array.shape[0] != target_num_points:
        final_point_cloud_array = np.resize(final_point_cloud_array, (target_num_points, 3))    
    assert final_point_cloud_array.shape[0] == target_num_points, f"Final point cloud array has incorrect shape:{final_point_cloud_array.shape}"
    return final_point_cloud_array

def process_tartan_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image for the tartan_drive dataset
    """
    img = ros_to_numpy(msg, output_resolution=IMAGE_SIZE) * 255
    img = img.astype(np.uint8)
    # reverse the axis order to get the image in the right orientation
    img = np.moveaxis(img, 0, -1)
    # convert rgb to bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    return img



def process_locobot_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image for the locobot dataset
    """
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = Image.fromarray(img)
    return pil_image


def process_scand_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/CompressedImage to a PIL image for the scand dataset
    """
    # convert sensor_msgs/CompressedImage to PIL image
    img = Image.open(io.BytesIO(msg.data))
    # center crop image to 4:3 aspect ratio
    w, h = img.size
    img = TF.center_crop(
        img, (h, int(h * IMAGE_ASPECT_RATIO))
    )  # crop to the right ratio
    # resize image to IMAGE_SIZE
    img = img.resize(IMAGE_SIZE)
    return img


############## Add custom image processing functions here #############

def process_sacson_img(msg) -> Image:
    np_arr = np.fromstring(msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_np)
    return pil_image


#######################################################################


def process_odom(
    odom_list: List,
    odom_process_func: Any,
    ang_offset: float = 0.0,
) -> Dict[np.ndarray, np.ndarray]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position and yaw
    """
    xys = []
    yaws = []
    for odom_msg in odom_list:
        xy, yaw = odom_process_func(odom_msg, ang_offset)
        xys.append(xy)
        yaws.append(yaw)
    return {"position": np.array(xys), "yaw": np.array(yaws)}


def nav_to_xy_yaw(odom_msg, ang_offset: float) -> Tuple[List[float], float]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position
    """

    position = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation
    yaw = (
        quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        + ang_offset
    )
    return [position.x, position.y], yaw


############ Add custom odometry processing functions here ############


#######################################################################

def get_images_lidar_and_odom(
    bag: rosbag.Bag,
    imtopics: List[str] or str,
    lidartopics: List[str] or str,  
    odomtopics: List[str] or str,
    img_process_func: Any,
    lidar_process_func: Any, 
    odom_process_func: Any,
    rate: float = 4.0,
    ang_offset: float = 0.0,
):
    """
    Get image, lidar, and odom data from a bag file

    Args:
        bag (rosbag.Bag): bag file
        imtopics (list[str] or str): topic name(s) for image data
        lidartopics (list[str] or str): topic name(s) for LiDAR data
        odomtopics (list[str] or str): topic name(s) for odom data
        img_process_func (Any): function to process image data
        lidar_process_func (Any): function to process lidar data
        odom_process_func (Any): function to process odom data
        rate (float, optional): rate to sample data. Defaults to 4.0.
        ang_offset (float, optional): angle offset to add to odom data. Defaults to 0.0.

    Returns:
        img_data (list): list of PIL images
        lidar_data (list): list of processed lidar data
        traj_data (list): list of odom data
       
    """
    # Check if bag has all the topics
    odomtopic = None
    imtopic = None
    lidartopic = None

    # Check image topic
    if type(imtopics) == str:
        imtopic = imtopics
    else:
        for imt in imtopics:
            if bag.get_message_count(imt) > 0:
                imtopic = imt
                break

    # Check odom topic
    if type(odomtopics) == str:
        odomtopic = odomtopics
    else:
        for ot in odomtopics:
            if bag.get_message_count(ot) > 0:
                odomtopic = ot
                break

    # Check lidar topic
    if type(lidartopics) == str:
        lidartopic = lidartopics
    else:
        for lt in lidartopics:
            if bag.get_message_count(lt) > 0:
                lidartopic = lt
                break

    if not (imtopic and odomtopic and lidartopic):
        # bag doesn't have all required topics
        return None, None, None

    synced_imdata = []
    synced_odomdata = []
    synced_lidar_data = []  
    currtime = bag.get_start_time()

    curr_imdata = None
    curr_odomdata = None
    curr_lidar_data = None  

    for topic, msg, t in bag.read_messages(topics=[imtopic, odomtopic, lidartopic]):
        if topic == imtopic:
            curr_imdata = msg
        elif topic == odomtopic:
            curr_odomdata = msg
        elif topic == lidartopic:
            curr_lidar_data = msg
        
        if (t.to_sec() - currtime) >= 1.0 / rate:
            if curr_imdata is not None and curr_odomdata is not None and curr_lidar_data is not None:
                synced_imdata.append(curr_imdata)
                synced_odomdata.append(curr_odomdata)
                synced_lidar_data.append(curr_lidar_data)
                currtime = t.to_sec()

    # 处理数据
    img_data = process_images(synced_imdata, img_process_func)
    traj_data = process_odom(
        synced_odomdata,
        odom_process_func,
        ang_offset=ang_offset,
    )

    lidar_data = process_lidar(synced_lidar_data,lidar_process_func)  
    return img_data, lidar_data, traj_data



def is_backwards(
    pos1: np.ndarray, yaw1: float, pos2: np.ndarray, eps: float = 1e-5
) -> bool:
    """
    Check if the trajectory is going backwards given the position and yaw of two points
    Args:
        pos1: position of the first point

    """
    dx, dy = pos2 - pos1
    return dx * np.cos(yaw1) + dy * np.sin(yaw1) < eps


# cut out non-positive velocity segments of the trajectory
def filter_backwards(
    img_list: List[Image.Image],
    lidar_data: List[np.ndarray],
    traj_data: Dict[str, np.ndarray],
    start_slack: int = 0,
    end_slack: int = 0,
) -> Tuple[List[np.ndarray],List[int]]:
    """
    Cut out non-positive velocity segments of the trajectory and include corresponding LiDAR data
    Args:
        img_list: list of images
        traj_data: dictionary of position and yaw data
        lidar_data: list of processed lidar data
        start_slack: number of points to ignore at the start of the trajectory
        end_slack: number of points to ignore at the end of the trajectory
    Returns:
        cut_trajs: list of cut trajectories
        start_times: list of start times of the cut trajectories
    """
    traj_pos = traj_data["position"]
    traj_yaws = traj_data["yaw"]
    cut_trajs = []
    start = True

    def process_pair(traj_pair: list) -> Tuple[List,List,Dict]:
        new_img_list, new_lidar_list,new_traj_data = zip(*traj_pair)
        new_traj_data = np.array(new_traj_data)
        new_traj_pos = new_traj_data[:, :2]
        new_traj_yaws = new_traj_data[:, 2]
        return (new_img_list,new_lidar_list, {"position": new_traj_pos, "yaw": new_traj_yaws})

    for i in range(max(start_slack, 1), len(traj_pos) - end_slack):
        pos1 = traj_pos[i - 1]
        yaw1 = traj_yaws[i - 1]
        pos2 = traj_pos[i]
        if not is_backwards(pos1, yaw1, pos2):
            if start:
                new_traj_pairs = [
                    (img_list[i - 1], lidar_data[i - 1],[*traj_pos[i - 1], traj_yaws[i - 1]])
                ]
                start = False
            elif i == len(traj_pos) - end_slack - 1:
                cut_trajs.append(process_pair(new_traj_pairs))
            else:
                new_traj_pairs.append(
                    (img_list[i - 1], lidar_data[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]])
                )
        elif not start:
            cut_trajs.append(process_pair(new_traj_pairs))
            start = True
    return cut_trajs


def quat_to_yaw(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Convert a batch quaternion into a yaw angle
    yaw is rotation around z in radians (counterclockwise)
    """
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw


def ros_to_numpy(
    msg, nchannels=3, empty_value=None, output_resolution=None, aggregate="none"
):
    """
    Convert a ROS image message to a numpy array
    """
    if output_resolution is None:
        output_resolution = (msg.width, msg.height)

    is_rgb = "8" in msg.encoding
    if is_rgb:
        data = np.frombuffer(msg.data, dtype=np.uint8).copy()
    else:
        data = np.frombuffer(msg.data, dtype=np.float32).copy()

    data = data.reshape(msg.height, msg.width, nchannels)

    if empty_value:
        mask = np.isclose(abs(data), empty_value)
        fill_value = np.percentile(data[~mask], 99)
        data[mask] = fill_value

    data = cv2.resize(
        data,
        dsize=(output_resolution[0], output_resolution[1]),
        interpolation=cv2.INTER_AREA,
    )

    if aggregate == "littleendian":
        data = sum([data[:, :, i] * (256**i) for i in range(nchannels)])
    elif aggregate == "bigendian":
        data = sum([data[:, :, -(i + 1)] * (256**i) for i in range(nchannels)])

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    else:
        data = np.moveaxis(data, 2, 0)  # Switch to channels-first

    if is_rgb:
        data = data.astype(np.float32) / (
            255.0 if aggregate == "none" else 255.0**nchannels
        )

    return data

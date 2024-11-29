import numpy as np
import io
import os
import rosbag
from PIL import Image
import cv2
from typing import Any, Tuple, List, Dict
import open3d as o3d
import pcl
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from pcl import PointCloud
import torchvision.transforms.functional as TF

IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = 4 / 3


def save_point_cloud_to_ply(point_cloud, output_path):
    """
    将点云保存为PLY格式
    :param point_cloud: 一个(N, 3)的numpy数组，包含点云坐标
    :param output_path: 输出路径，保存为PLY文件
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


def process_lidar(msg) -> PointCloud:
    """
    Process LiDAR data from a topic that publishes sensor_msgs/PointCloud2 to a PointCloud object.
    This function will convert the PointCloud2 message to a PointCloud object and return it.

    Args:
        msg (sensor_msgs/PointCloud2): The ROS PointCloud2 message

    Returns:
        pcl.PointCloud: The processed PointCloud object
    """
    # Convert PointCloud2 message to numpy array
    pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    pc_np = np.array(list(pc_data))

    # Convert to pcl.PointCloud format
    pcl_cloud = pcl.PointCloud()
    pcl_cloud.from_array(pc_np.astype(np.float32))

    return pcl_cloud

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
    imtopics: List[str],
    odomtopics: List[str],
    lidartopics: List[str],  # LiDAR 数据话题
    img_process_func: Any,
    odom_process_func: Any,
    lidar_process_func: Any,  # LiDAR 数据处理函数
    rate: float = 4.0,
    ang_offset: float = 0.0,
):
    """
    Get image, lidar, and odom data from a bag file

    Args:
        bag (rosbag.Bag): bag file
        imtopics (list[str] or str): topic name(s) for image data
        odomtopics (list[str] or str): topic name(s) for odom data
        lidartopics (list[str] or str): topic name(s) for LiDAR data
        img_process_func (Any): function to process image data
        odom_process_func (Any): function to process odom data
        lidar_process_func (Any): function to process lidar data
        rate (float, optional): rate to sample data. Defaults to 4.0.
        ang_offset (float, optional): angle offset to add to odom data. Defaults to 0.0.

    Returns:
        img_data (list): list of PIL images
        traj_data (list): list of odom data
        lidar_data (list): list of processed lidar data
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
    synced_lidar_data = []  # 新增：同步的LiDAR数据
    currtime = bag.get_start_time()

    curr_imdata = None
    curr_odomdata = None
    curr_lidar_data = None  # 新增：LiDAR数据

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
    lidar_data = process_lidar(synced_lidar_data, lidar_process_func)  # 新增：处理LiDAR数据

    return img_data, traj_data, lidar_data



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
    traj_data: Dict[str, np.ndarray],
    lidar_data: List[pcl.PointCloud],
    start_slack: int = 0,
    end_slack: int = 0,
) -> Tuple[List[np.ndarray], List[int]]:
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

    def process_pair(traj_pair: list) -> Tuple[List, Dict, List]:
        new_img_list, new_traj_data, new_lidar_data = zip(*traj_pair)
        new_traj_data = np.array(new_traj_data)
        new_traj_pos = new_traj_data[:, :2]
        new_traj_yaws = new_traj_data[:, 2]
        return (new_img_list, {"position": new_traj_pos, "yaw": new_traj_yaws}, list(new_lidar_data))

    for i in range(max(start_slack, 1), len(traj_pos) - end_slack):
        pos1 = traj_pos[i - 1]
        yaw1 = traj_yaws[i - 1]
        pos2 = traj_pos[i]
        if not is_backwards(pos1, yaw1, pos2):
            if start:
                new_traj_pairs = [
                    (img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]], lidar_data[i - 1])
                ]
                start = False
            elif i == len(traj_pos) - end_slack - 1:
                cut_trajs.append(process_pair(new_traj_pairs))
            else:
                new_traj_pairs.append(
                    (img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]], lidar_data[i - 1])
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

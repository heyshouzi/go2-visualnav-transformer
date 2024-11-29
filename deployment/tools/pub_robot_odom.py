import sys
import rospy
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ # 宇树SDK的Odometry数据类型
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, Quaternion
import tf

class odom_pub:
    def __init__(self, odom_topic='/robot_odom'):
        rospy.init_node("odom_pub")
        self.odom_pub = rospy.Publisher(odom_topic, Odometry, queue_size=10)
        self.latest_odom = None
        # 初始化SDK连接
        if len(sys.argv) > 1:
            ChannelFactoryInitialize(0, sys.argv[1])
        else:
            ChannelFactoryInitialize(0)

        
        self.timer  = rospy.Timer(rospy.Duration(1),self.print_odom)

        # 订阅SDK的odometry数据
        self.sub = ChannelSubscriber("rt/utlidar/robot_odom", Odometry_)

        # 设定定时器或者回调函数来处理接收到的消息
        self.sub.Init(self.odomHandler,10)


    def odomHandler(self, msg: Odometry_):


        # 从SDK的Odometry_数据中提取信息
        odom_ros = Odometry()

        # 设置时间戳和框架
        odom_ros.header.stamp = rospy.Time.now()
        odom_ros.header.frame_id = "odom"
        odom_ros.child_frame_id = msg.child_frame_id

        # 转换Pose数据
        odom_ros.pose.pose = Pose()
        odom_ros.pose.pose.position.x = msg.pose.pose.position.x
        odom_ros.pose.pose.position.y = msg.pose.pose.position.y
        odom_ros.pose.pose.position.z = msg.pose.pose.position.z
        # 注意：SDK中的旋转数据一般为四元数
        odom_ros.pose.pose.orientation = Quaternion(
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
        )

        # 转换Twist数据
        odom_ros.twist.twist = Twist()
        odom_ros.twist.twist.linear.x = msg.twist.twist.linear.x
        odom_ros.twist.twist.linear.y = msg.twist.twist.linear.y
        odom_ros.twist.twist.linear.z = msg.twist.twist.linear.z
        odom_ros.twist.twist.angular.x = msg.twist.twist.angular.x
        odom_ros.twist.twist.angular.y = msg.twist.twist.angular.y
        odom_ros.twist.twist.angular.z = msg.twist.twist.angular.z

        # 发布Odometry消息
        self.odom_pub.publish(odom_ros)
        self.latest_odom = odom_ros
    
    def print_odom(self,event):
        if self.latest_odom:
            rospy.loginfo(f"x:{self.latest_odom.pose.pose.position.x}, y:{self.latest_odom.pose.pose.position.y},z: {self.latest_odom.pose.pose.position.z}")
            rospy.loginfo(f"vx:{self.latest_odom.twist.twist.linear.x}, vy:{self.latest_odom.twist.twist.linear.y},vz: {self.latest_odom.twist.twist.linear.z}")

if __name__ == "__main__":
    # 创建Odometry发布者对象
    odom_publisher = odom_pub()

    # 开始ROS消息循环
    rospy.spin()


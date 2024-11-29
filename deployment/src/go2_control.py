# ROS
import rospy
from geometry_msgs.msg import Twist
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.core.channel import  ChannelFactoryInitialize
import time 

import yaml

# CONSTS
CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
VEL_TOPIC = robot_config["vel_navi_topic"]
RATE = 9

class RobotController:
    
    def __init__(self):
        self.client = SportClient()  # 创建运动控制客户端
        self.client.SetTimeout(10.0)
        self.client.Init()

        self.vx = 0  # 线速度
        self.vy = 0  # 侧向速度
        self.vyaw = 0  # 角速度

        # 初始化 ROS 节点
        rospy.init_node('robot_controller', anonymous=True)

        # 订阅速度话题
        rospy.Subscriber(VEL_TOPIC, Twist, self.velocity_callback)

    def velocity_callback(self, msg):
        # 从 Twist 消息中提取速度
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.vyaw = msg.angular.z

    def move_robot(self):
        rate = rospy.Rate(RATE)  
        while not rospy.is_shutdown():
            # 调用 SDK 的运动控制
            print(f"go2 vx:{self.vx}, vy:{self.vy}, vyaw:{self.vyaw}")
            self.client.Move(self.vx, self.vy, self.vyaw)
            rate.sleep()

    def stop_robot(self):
        print("Stopping robot by user (ctrl + c)...")
        self.client.StopMove()

if __name__ == '__main__':
    ChannelFactoryInitialize(0)
    controller = RobotController()
    try:    
        controller.move_robot()
    except KeyboardInterrupt:
        controller.stop_robot()
    
    except rospy.ROSInterruptException:
        pass
    
        

import rospy
from geometry_msgs.msg import Twist
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
import yaml

CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
VEL_TOPIC = robot_config["vel_navi_topic"]


class RobotController:
    def __init__(self):
        self.client = SportClient()  # 创建运动控制客户端
        self.client.SetTimeout(10.0)
        self.client.Init()

        self.vx = 0  # 线速度
        self.vy = 0  # 侧向速度
        self.vyaw = 0  # 角速度

        # 初始化 ROS 节点
        rospy.init_node('go2_controller')

        # 订阅速度话题
        rospy.Subscriber(VEL_TOPIC, Twist, self.velocity_callback)

    def velocity_callback(self, msg):
        # 从 Twist 消息中提取速度
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.vyaw = msg.angular.z

    def move_robot(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            # 调用 SDK 的运动控制
            self.client.Move(self.vx, self.vy, self.vyaw)
            rate.sleep()

    def stop_robot(self):
        """在程序停止时停止机器人运动"""
        print("Stopping robot...")
        self.client.StopMove()  # 停止机器人

if __name__ == "__main__":
    ChannelFactoryInitialize(0)
    controller = RobotController()
    try:
        controller.move_robot()  # 开始机器人运动
    except KeyboardInterrupt:
        # 捕获 Ctrl+C 信号，停止机器人并退出
        controller.stop_robot()  # 停止机器人
        print("\nProgram terminated by user (Ctrl + C).")
    except rospy.ROSInterruptException:
        pass

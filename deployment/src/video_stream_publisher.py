import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
import cv2

from topic_names import IMAGE_TOPIC 



class VideoStreamPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('video_stream_publisher')
        
        # Initialize CvBridge to convert OpenCV images to ROS Image messages
        self.bridge = CvBridge()

        # Create a publisher for the video feed
        self.image_pub = rospy.Publisher(IMAGE_TOPIC, Image, queue_size=10)
        
        # Set up the video client
        if len(sys.argv) > 1:
            ChannelFactoryInitialize(0, sys.argv[1])
        else:
            ChannelFactoryInitialize(0)

        self.client = VideoClient()  # Create a video client
        self.client.SetTimeout(3.0)
        self.client.Init()

        self.code, self.data = self.client.GetImageSample()

    def get_and_publish_image(self):
        # Request normal when code == 0
        while self.code == 0:
            # Get Image data from Go2 robot
            self.code, self.data = self.client.GetImageSample()

            # Convert to numpy image
            image_data = np.frombuffer(bytes(self.data), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            if image is not None:
                try:
                    # Convert OpenCV image to ROS Image message
                    ros_image = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
                    
                    # Publish the ROS Image message
                    self.image_pub.publish(ros_image)
                except Exception as e:
                    rospy.logerr("Error converting image: %s", str(e))

            # Sleep to ensure we're not overwhelming the system
            rospy.sleep(0.1)

        if self.code != 0:
            rospy.logerr("Get image sample error. code: %d", self.code)
        else:
            # Optionally capture an image if needed
            cv2.imwrite("front_image.jpg", image)

    def run(self):
        # Start publishing the video stream
        rospy.loginfo("Starting video stream publisher...")
        self.get_and_publish_image()


if __name__ == "__main__":
    # Create a VideoStreamPublisher object and start it
    video_stream_publisher = VideoStreamPublisher()
    video_stream_publisher.run()

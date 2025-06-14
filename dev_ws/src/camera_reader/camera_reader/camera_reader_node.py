#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class CameraReaderNode(Node):
    def __init__(self):
        super().__init__('camera_reader_node')
        self.publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # Publish at 10 Hz
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)  # Open the default camera

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera")
            rclpy.shutdown()

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher.publish(msg)
        else:
            self.get_logger().warning("Failed to capture frame from camera")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraReaderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


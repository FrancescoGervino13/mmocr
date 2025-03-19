import rclpy
from rclpy.node import Node
import cv2
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_folder_publisher')
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.timer = self.create_timer(1.0, self.publish_image)  # Publish at 1Hz
        self.bridge = CvBridge()

        # Set the folder containing images
        self.image_folder = '/home/user/images/'  # Change this path!
        self.image_files = sorted([f for f in os.listdir(self.image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if not self.image_files:
            self.get_logger().error('No images found in the folder!')
            rclpy.shutdown()

        self.index = 0  # Start from the first image

    def publish_image(self):
        if self.index >= len(self.image_files):
            self.index = 0  # Loop back to the first image

        image_path = os.path.join(self.image_folder, self.image_files[self.index])
        frame = cv2.imread(image_path)

        if frame is None:
            self.get_logger().warn(f'Could not read image {image_path}')
            return

        # Convert OpenCV image to ROS2 Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published image: {self.image_files[self.index]}')

        self.index += 1

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
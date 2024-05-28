import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class CamNode(Node):

    def __init__(self):
        super().__init__('cam_node')
        self.publisher_ = self.create_publisher(String, 'TOPIC', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = ""
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    cam_node = CamNode()

    rclpy.spin(cam_node)

    cam_node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
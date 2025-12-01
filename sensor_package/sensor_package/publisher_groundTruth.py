import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import re

class LinePublisher(Node):
    def __init__(self):
        super().__init__('line_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'groundT_topic', 10)

        self.file_path = '/home/lazaros/Desktop/rwr/TaskX/23-11-25/r0/r0a5.log'
        self.timer = self.create_timer(0.1, self.publish_lines_once)
        self.lines = None
        self.index = 0

    def publish_lines_once(self):
        # Load file on first timer callback
        if self.lines is None:
            try:
                with open(self.file_path, 'r') as f:
                    self.lines = f.readlines()
                self.get_logger().info("File loaded, starting line-by-line publishing...")
            except Exception as e:
                self.get_logger().error(f"Error reading file: {e}")
                self.lines = []
        
        if self.index < len(self.lines):
            original = self.lines[self.index].rstrip('\n')

            # --- Extract first number after '>' ---
            match = re.search(r'>\s*[^\d-]*(-?\d+)', original)
            if match:
                num = float(match.group(1))
            else:
                num = 0.0  # fallback or skip

            # Publish as single-value Float32MultiArray
            msg = Float32MultiArray()
            msg.data = [num]
            self.publisher_.publish(msg)

            self.get_logger().info(f"Published: {msg.data}")

            self.index += 1
        else:
            self.get_logger().info("Finished publishing all lines.")
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = LinePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
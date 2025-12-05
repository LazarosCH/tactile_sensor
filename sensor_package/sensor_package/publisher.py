import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import time
import serial
import numpy as np


class LinePublisher(Node):
    def __init__(self):
        super().__init__('line_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'line_topic', 10)
        self.publisher_gt = self.create_publisher(Float32MultiArray, 'groundT_topic', 10)

        self.declare_parameter('sensor_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 19200)
        sensor_port = self.get_parameter('sensor_port').get_parameter_value().string_value
        baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value

        self.ser = serial.Serial(sensor_port, baud_rate)

        self.timer = self.create_timer(0.05, self.publish_sensor_readings)
        self.lines = None
        self.index = 0

    def publish_sensor_readings(self):
        msg = Float32MultiArray()
        msg_gt = Float32MultiArray()

        # line = self.ser.readline().decode().strip()
        line = self.ser.readline().decode('ascii', errors='ignore').strip()
        # line_split = line.split(",")
        if line:
            try:
                # msg.data = [int(x) for x in line.split(",")]
                msg.data = [float(x) for x in line.split(",")]
                # for i, val in enumerate(msg.data):
                    # print(f"Value {i}: {val}")
                # TODO: remove class
                msg.data.pop(0)
                msg_gt.data = [msg.data.pop(0)] # weight

                # sum = np.sum( msg.data)
                # for i in range(len(msg.data)):
                #     msg.data[i] /= sum


            except ValueError:
                print("⚠️ Invalid data received:", line)
        self.publisher_.publish(msg)
        self.publisher_gt.publish(msg_gt)
        self.ser.reset_input_buffer()  # optional: clear remaining old data

    # def publish_lines_once(self):
    #     # Load file on first timer callback
    #     if self.lines is None:
    #         try:
    #             with open(self.file_path, 'r') as f:
    #                 self.lines = f.readlines()
    #             self.get_logger().info("File loaded, starting line-by-line publishing...")
    #         except Exception as e:
    #             self.get_logger().error(f"Error reading file: {e}")
    #             self.lines = []
        
    #     if self.index < len(self.lines):
    #         original = self.lines[self.index].rstrip('\n')

    #         # --- Extract text after first comma ---
    #         if ',' in original:
    #             processed = original.split(',', 1)[1]   # keep only after first comma
    #         else:
    #             processed = ""  # or: procepublisher_ssed = original

    #         float_array = [float(x) for x in processed.split(',')]

    #         msg = Float32MultiArray()
    #         msg.data = float_array
    #         self.publisher_.publish(msg)

    #         self.get_logger().info(f"Published: {msg.data}")

    #         self.index += 1
    #     else:
    #         self.get_logger().info("Finished publishing all lines.")
    #         self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = LinePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



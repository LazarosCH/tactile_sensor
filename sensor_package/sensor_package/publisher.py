import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import time
import serial
import numpy as np

NUM_SAMPLES = 5
NUM_DIM = 15

class LinePublisher(Node):
    array : np.ndarray
    def __init__(self):
        super().__init__('line_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'raw_signal_topic', 10)
        self.publisher_gt = self.create_publisher(Float32MultiArray, 'groundT_topic', 10)

        self.declare_parameter('sensor_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 57600)
        sensor_port = self.get_parameter('sensor_port').get_parameter_value().string_value
        baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value

        self.ser = serial.Serial(sensor_port, baud_rate)

        self.timer = self.create_timer(0.02, self.publish_sensor_readings)
        self.lines = None
        self.index = 0

        self.array = np.zeros((NUM_SAMPLES,6))
        self.init = False

    def publish_sensor_readings(self):
        msg = Float32MultiArray()
        msg_gt = Float32MultiArray()

        line = self.ser.readline().decode('ascii', errors='ignore').strip()
        if line:
            try:
                msg.data = [float(x) for x in line.split(",")]
                msg_gt.data = [msg.data.pop(0)] # weight
                msg_gt.data.append(msg.data[0]) # weight


            except ValueError:
                print("⚠️ Invalid data received:", line)
                return

        print(len(msg.data))
        if len(msg.data) < NUM_DIM:
            self.publisher_gt.publish(msg_gt)
            return

        self.publisher_.publish(msg)
        self.publisher_gt.publish(msg_gt)
        self.ser.reset_input_buffer()  # optional: clear remaining old data


def main(args=None):
    rclpy.init(args=args)
    node = LinePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



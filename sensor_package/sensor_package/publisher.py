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
        self.publisher_ = self.create_publisher(Float32MultiArray, 'line_topic', 10)
        self.publisher_new = self.create_publisher(Float32MultiArray, 'new_line_topic', 10)
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
        msg_new = Float32MultiArray()
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
                # msg.data.pop(0)
                msg_gt.data = [msg.data.pop(0)] # weight
                msg_gt.data.append(msg.data[0]) # weight

                # sum = np.sum( msg.data)
                # for i in range(len(msg.data)):
                #     msg.data[i] /= sum

            except ValueError:
                print("⚠️ Invalid data received:", line)
                return

        print(len(msg.data))
        if len(msg.data) < NUM_DIM:
            return
        
        # if not self.init:
        #     self.init = True
        #     for i in range(6):
        #         for j in range(NUM_SAMPLES):
        #             self.array[j,i] = msg.data[i]
        
        # rotated = np.roll(self.array,shift=1, axis=0)
        # # print(rotated.shape)
        # # print(len(msg.data))
        # msg_new.data = [0.0,0.0,0.0,0.0,0.0,0.0]
        # for i in range(6):
        #         rotated[0,i] = msg.data[i]

        # avg_values = np.average(rotated,axis = 0)   
        # # print(avg_values)

        # for i in range(6):
        #     msg_new.data[i] = avg_values[i]

        # self.array = rotated

        # if len(msg.data) == 15:
        self.publisher_.publish(msg)
        # print(msg.data[0])
        # self.publisher_new.publish(msg_new)
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



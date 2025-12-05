import keras 
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import pickle
import joblib

USE_LSTM = False

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'line_topic',
            self.listener_callback,
            10
        )

        self.publisher_ = self.create_publisher(
            Float32MultiArray,
            'touch_propability_topic',  
            10
        )

        self.buffer = []
        self.model = keras.models.load_model(f"/home/papaveneti/ros_ws/src/sensor_package/sensor_package/config/model_position.keras")
        self.x_scaler = joblib.load(f"/home/papaveneti/ros_ws/src/sensor_package/sensor_package/config/x_scaler_position.save")
        self.time_steps = 15


    def listener_callback(self, msg):
        data = list(msg.data)
        self.buffer.append(data)

        print(data)

        
        if USE_LSTM:
            if len(self.buffer) > self.time_steps:
                self.buffer.pop(0)
                #self.buffer.pop(-1)
            if len(self.buffer) == self.time_steps:
                stacked_ = np.array(self.buffer)  # shape = (time_steps, 12)
                Dstacked = stacked_ - np.mean(stacked, axis=0)
                stacked = np.concatenate([stacked_, Dstacked], axis=1)


                X_scaled = self.x_scaler.transform(stacked.reshape(-1,12))
                X_scaled = X_scaled.reshape(1, self.time_steps, 12)
                y_pred = self.model.predict(X_scaled)

                # if np.max(y_pred)>0.9:
                    # if np.argmax(y_pred)>0:
                        # self.get_logger().info(f"touch{np.argmax(y_pred)}")
                self.get_logger().info(f"ypred={y_pred}\n")

                msg_out = Float32MultiArray()
                msg_out.data = y_pred.tolist()  # Flatten in case y_pred is multi-dimensional
                self.publisher_.publish(msg_out)
        else:
            data = np.array(data).reshape(1, -1)
            print(data)
            X_scaled = self.x_scaler.transform(data)
            y_pred = self.model.predict(X_scaled)
            # if np.max(y_pred)>0.8:
                # self.get_logger().info(f"touch{np.argmax(y_pred)}")
            msg_out = Float32MultiArray()
            msg_out.data = y_pred[0].tolist()  # Flatten in case y_pred is multi-dimensional
            self.publisher_.publish(msg_out)


            

            

def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

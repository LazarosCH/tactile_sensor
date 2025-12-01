import keras 
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import pickle
import joblib
import tensorflow as tf

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription1 = self.create_subscription(
            Float32MultiArray,
            'line_topic',
            self.listener_callback1,
            10
        )

        self.publisher_ = self.create_publisher(
            Float32MultiArray,
            'force_array',  
            10
        )

        self.Position=0
        self.buffer = []
        self.model_list = []
        self.x_scaler_list = []
        self.y_scaler_list = []

 
        model_path = f"/home/lazaros/Desktop/rwr/TaskX/Models/FORCE_ARRAY_save/model_array.keras"
        x_scaler_path = f"/home/lazaros/Desktop/rwr/TaskX/Models/FORCE_ARRAY_save/x_scaler_array.save" 
            
        self.model = keras.models.load_model(model_path, compile=False)
        self.x_scaler = joblib.load(x_scaler_path)


    def listener_callback1(self, msg):
        data = list(msg.data)
        self.buffer.append(data)
        
        if len(self.buffer) > 5:
            self.buffer.pop(0)
        if len(self.buffer) == 5:
            stacked = np.array(self.buffer)  # shape = (3, 12)
            
            
            X_scaled = self.x_scaler.transform(stacked)
            X_scaled = np.expand_dims(X_scaled, axis=0)

            y_scaled_pred = self.model.predict(X_scaled)

            msg_out = Float32MultiArray()
            msg_out.data = y_scaled_pred.flatten().tolist()  # <-- convert to 1D Python list
            self.publisher_.publish(msg_out)
            self.get_logger().info(f"{msg_out}")


def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

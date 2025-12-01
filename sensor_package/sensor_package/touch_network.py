import keras 
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import pickle
import joblib

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
        self.model = keras.models.load_model(f"/home/lazaros/Desktop/rwr/TaskX/Models/TOUCH/model_position.keras")
        self.x_scaler = joblib.load(f"/home/lazaros/Desktop/rwr/TaskX/Models/TOUCH/x_scaler_position.save")



    def listener_callback(self, msg):
        data = list(msg.data)
        self.buffer.append(data)
        
        if len(self.buffer) > 5:
            self.buffer.pop(0)
        if len(self.buffer) == 5:
            stacked = np.array(self.buffer)  # shape = (3, 12)
            X_scaled = self.x_scaler.transform(stacked)
            X_scaled = np.expand_dims(X_scaled, axis=0)
            y_pred = self.model.predict(X_scaled)
            self.get_logger().info(f"ypred={y_pred}\n")

            msg_out = Float32MultiArray()
            msg_out.data = y_pred.flatten().tolist()  # Flatten in case y_pred is multi-dimensional
            self.publisher_.publish(msg_out)
        

def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

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
        self.subscription1 = self.create_subscription(
            Float32MultiArray,
            'line_topic',
            self.listener_callback1,
            10
        )
        self.subscription2 = self.create_subscription(
            Float32MultiArray,
            'position_propability_topic',
            self.listener_callback2,
            10
        )
        self.publisher_ = self.create_publisher(
            Float32MultiArray,
            'force_topic',  
            10
        )

        self.Position=0
        self.buffer = []
        self.model_list = []
        self.x_scaler_list = []
        self.y_scaler_list = []

        for Number in range(1, 15):  
            model_path = f"/home/lazaros/Desktop/rwr/TaskX/Models/Force{Number}/model{Number}.keras"
            x_scaler_path = f"/home/lazaros/Desktop/rwr/TaskX/Models/Force{Number}/x_scaler{Number}.save"
            y_scaler_path = f"/home/lazaros/Desktop/rwr/TaskX/Models/Force{Number}/y_scaler{Number}.save"
            
            model = keras.models.load_model(model_path)
            x_scaler = joblib.load(x_scaler_path)
            y_scaler = joblib.load(y_scaler_path)
            
            self.model_list.append(model)
            self.x_scaler_list.append(x_scaler)
            self.y_scaler_list.append(y_scaler)


    def listener_callback2(self, msg1):
        data = np.array(msg1.data)
        self.Position = np.argmax(data)


    def listener_callback1(self, msg):
        data = list(msg.data)
        self.buffer.append(data)
        
        if len(self.buffer) > 5:
            self.buffer.pop(0)
        if len(self.buffer) == 5:
            stacked = np.array(self.buffer)  # shape = (3, 12)
            
            if self.Position>0:
                X_scaled = self.x_scaler_list[self.Position-1].transform(stacked)
                X_scaled = np.expand_dims(X_scaled, axis=0)
                y_scaled_pred = self.model_list[self.Position-1].predict(X_scaled)
                y_pred = self.y_scaler_list[self.Position-1].inverse_transform(y_scaled_pred)
            else:
                y_pred= [0]
            self.get_logger().info(f"ypred={y_pred}\n")
            msg = Float32MultiArray()
            msg.data = y_pred
            self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

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
            'raw_signal_topic',
            self.listener_callback,
            1
        )

        self.publisher_ = self.create_publisher(
            Float32MultiArray,
            'force_estimation_topic',  
            1
        )

        self.Scale = 550
        self.buffer = []
        self.model = keras.models.load_model(f"/home/papaveneti/ros_ws/src/sensor_package/sensor_package/config/combined_model.keras")
        self.time_steps = 10
        self.dims = 15
        self.scalers = []
        for i in range(15):
            self.scalers.append(joblib.load(f"/home/papaveneti/ros_ws/src/sensor_package/sensor_package/config/Scalers/scaler{i}.save"))
        

        # ========= Weight Factors ==========

        # self.Ki  = [ -365.80005192,    32.88599582,  1053.3719454,    173.34400912,
        #             -186.78379119,   204.89717288,   -80.34749169,  1314.43792953,
        #             354.05688713,  -129.41823737,  -764.96428362,   -66.1146445,
        #             -2737.52862746,  3170.29413826,  -526.19411382]
        
        self.Ki = np.ones(15)
        # self.Ki[1] = 1.0
        # self.Ki[2] = 0.2
        # self.Ki[3] = 1.0

        # self.Ki = [1, 2.        , 0.59507752, 1.50657813, 1,
        #            1, 1.36385654, 1.84554181, 1.86682171, 1,
        #            1, 2.        , 1.33556953, 1.87664483, 1]
        

        # temp
        self.Xmean= np.zeros(15)
        # self.Xmean[1] = 0.11561702
        # self.Xmean[2] = 0.24170786
        # self.Xmean[3] = 0.17270825
        # self.Xmean= [0, 0.11561702, 0.24170786, 0.17270825, 0,
        #              0, 0.20878498, 0.19127495, 0.07142669, 0,
        #              0, 0.19066084, 0.23498102, 0.25118775, 0]
        

    def listener_callback(self, msg):
        data = list(msg.data)
        self.buffer.append(data)

        print(data)

        if len(self.buffer) > self.time_steps:
            self.buffer.pop(0)
        if len(self.buffer) == self.time_steps:
            stacked_ = np.array(self.buffer)  # shape = (time_steps, 12)

            X_inputs = []
            for i in range(self.dims):
                scaled_col = self.scalers[i].transform(stacked_[:, i].reshape(-1, 1))
                X_inputs.append(scaled_col.reshape(-1, self.time_steps, 1))

            y_pred = self.model.predict(X_inputs)
            # y_pred_scaled = [float(y[0][0])*self.Scale for y in y_pred]
            y_pred_scaled = [(float(y[0][0]) -xm) * k * self.Scale for y, k, xm in zip(y_pred, self.Ki, self.Xmean)]

            # Mageirema
            y_pred_scaled[9] = (y_pred_scaled[4] + y_pred_scaled[14])/2.0

            msg_out = Float32MultiArray()
            msg_out.data = y_pred_scaled  # Flatten in case y_pred is multi-dimensional
            self.publisher_.publish(msg_out)            

            

def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

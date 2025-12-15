import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import pickle
import matplotlib
matplotlib.use('Qt5Agg')  # must be before pyplot
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        
        self.subscription1 = self.create_subscription(Float32MultiArray, 'force_estimation_topic', self.listener_callback1, 1)
        self.subscription2 = self.create_subscription(Float32MultiArray, 'groundT_topic', self.listener_callback3, 1)
        
        path_areas="/home/papaveneti/ros_ws/src/sensor_package/sensor_package/Visualization/Patches.pkl"
        with open(path_areas, 'rb') as f:
            self.patches = pickle.load(f)
        path_fingers="/home/papaveneti/ros_ws/src/sensor_package/sensor_package/Visualization/Fingers.pkl"
        with open(path_fingers, 'rb') as f:
            self.fingers = pickle.load(f)
        path_wires="/home/papaveneti/ros_ws/src/sensor_package/sensor_package/Visualization/Wires.pkl"
        with open(path_wires, 'rb') as f:
            self.wires = pickle.load(f)   
        
        self.Forces = np.zeros(15)
        self.GT =0.0

        # Flags to track fresh data
        self.new_Forces  = False
        self.new_gt = False
        self.scale_force = 550
        self.samples = 50

        plt.ion()
        self.fig, (self.ax, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.ax.set_ylim(-0.6, 5.5)
        self.ax.axis('equal')
        self.ax.axis('off')

        self.time_history = []
        self.force_history = []
        self.force_history_l = []
        self.force_history_r = []
        self.gt_history = []
        self.t = 0.0
        self.sum = 0.0
        self.sum_r = 0.0
        self.sum_l = 0.0
        self.thresholds = [20.,20.,20.,20.,50.,
                           20.,20.,20.,20.,50.,
                           25.,20.,20.,50.,20.]
        self.power_factor = [0.6,0.6,0.4,0.6,0.6,
                             0.6,0.6,0.6,0.6,0.6,
                             0.6,0.6,0.6,0.6,0.6,]
        self.force_factor = 9.81/1000.0


    def listener_callback3(self, msg):
        self.GT = np.array(msg.data)[0]*self.force_factor 
        self.new_gt = True

    def listener_callback1(self, msg):
        self.Forces = np.array(msg.data)
        for i in range(15):
            if self.Forces[i] < self.thresholds[i]:
                self.Forces[i] = 0.0
        self.sum = np.sum(self.Forces[[1,2,3,6,7,8,11,12,13]])*self.force_factor 
        self.sum_r = np.sum(self.Forces[[0,5,10]])*self.force_factor 
        self.sum_l = np.sum(self.Forces[[4,9,14]])*self.force_factor 
        self.new_Forces = True

    def update_plot(self):
        # Only update if all three subscribers have fresh data
        if not (self.new_Forces and self.new_gt):
            return

        self.ax.cla()
        self.ax.set_ylim(-0.6, 5.5)
        self.ax.axis('equal')
        self.ax.axis('off')

        print(len(self.patches))

        # Draw datasets with updated alpha
        for i, (xi, yi) in enumerate(self.patches):
            self.ax.plot(xi, yi, color='black', linewidth=1)
            if self.Forces[i] > self.thresholds[i]:
                alpha = float(np.clip(np.power( self.Forces[i]/self.scale_force, self.power_factor[i]) , 0.0, 1.0))
                self.ax.fill(xi, yi, color='red', alpha=alpha)
            else:
                self.ax.fill(xi, yi, color='green', alpha=1.0)

        self.ax.text(-0.35, 4.2, f"F={self.sum :.1f}N", fontsize=16, color='black')
        self.ax.text(-2.25, 4.2, f"F={self.sum_l :.1f}N", fontsize=16, color='black')
        self.ax.text( 1.55, 4.2, f"F={self.sum_r :.1f}N", fontsize=16, color='black')

        for i, (xi, yi) in enumerate(self.fingers):
            self.ax.plot(xi, yi, color='black', linewidth=1.5)
        for i, (xi, yi) in enumerate(self.wires):
            self.ax.plot(xi, yi, color='black', linewidth=1.5)

        # -------- RIGHT PLOT: F vs t --------
        self.ax2.cla()
        self.time_history.append(self.t)
        self.force_history.append(self.sum)
        self.force_history_l.append(self.sum_l)
        self.force_history_r.append(self.sum_r)
        self.gt_history.append(self.GT)
        self.t += 0.1  # loop step

        if len(self.time_history) > self.samples:
            self.time_history = self.time_history[-self.samples:]
            self.force_history = self.force_history[-self.samples:]
            self.force_history_l = self.force_history_l[-self.samples:]
            self.force_history_r = self.force_history_r[-self.samples:]
            self.gt_history = self.gt_history[-self.samples:]

        self.ax2.plot(self.time_history, self.force_history , linewidth=2, label="Front Face Force")
        # self.ax2.plot(self.time_history, self.force_history_l, linewidth=2, label="Left Face Force")
        # self.ax2.plot(self.time_history, self.force_history_r, linewidth=2, label="Right Face Force")
        self.ax2.plot(self.time_history, self.gt_history , linewidth=2, linestyle="--", label="Ground Truth")
        self.ax2.axhline(y=500*self.force_factor , color="red", linestyle="--", linewidth=2,  label="Load Cell Limit / Node limit")
        self.ax2.set_title("Force vs Time")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Force (N)")
        self.ax2.grid(True)
        self.ax2.legend(loc="upper right", frameon=True)
        self.ax2.set_ylim([0,650.0*self.force_factor ])

        # redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Reset flags
        self.new_force = False
        self.new_gt = False
        self.new_position = False


def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.update_plot()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close('all')


if __name__ == '__main__':
    main()

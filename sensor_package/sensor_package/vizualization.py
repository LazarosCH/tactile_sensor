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
        
        self.subscription3 = self.create_subscription(Float32MultiArray, 'groundT_topic', self.listener_callback3, 10)
        self.subscription2 = self.create_subscription(Float32MultiArray, 'force_topic', self.listener_callback2, 10)
        self.subscription1 = self.create_subscription(Float32MultiArray, 'position_propability_topic', self.listener_callback1, 10)
        
        path_areas="/home/lazaros/Desktop/rwr/Graphics/AreasTip.pkl"
        with open(path_areas, 'rb') as f:
            self.datasets = pickle.load(f)
        
        self.x= np.array([0.0, -0.4, 0.4, 0.0, -0.6, 0.6, -0.7, 0.7, -1.4, -1.7, -1.7, 1.4, 1.7, 1.7])
        self.y= np.array([2.7, 2.1, 2.1, 1.4, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0])
        
        self.propabilities = np.zeros(len(self.x)+1)
        self.Force = 0.0
        self.GT =0.0
        self.position = (0.0, 0.0)
        self.Position1=0

        # Flags to track fresh data
        self.new_force = False
        self.new_gt = False
        self.new_position = False
        self.maxvalue = 1000
        self.samples = 50

        plt.ion()
        self.fig, (self.ax, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.ax.set_ylim(-0.6, 4)
        self.ax.axis('equal')
        self.ax.axis('off')

        self.time_history = []
        self.force_history = []
        self.gt_history = []
        self.t = 0.0

    def listener_callback3(self, msg):
        self.GT = np.array(msg.data)[0]
        self.new_gt = True

    def listener_callback2(self, msg):
        self.Force = np.clip(np.array(msg.data)[0], 0.0, self.maxvalue)
        self.new_force = True

    def listener_callback1(self, msg):
        self.propabilities = np.array(msg.data)
        self.Position1 = np.argmax(self.propabilities)
        self.compute_exactposition()
        self.new_position = True

    def compute_exactposition(self):
        if self.propabilities[0]<0.7:
            xi = np.sum(self.propabilities[1:] * self.x)
            yi = np.sum(self.propabilities[1:] * self.y)
            self.position = (xi, yi)

    def update_plot(self):
        # Only update if all three subscribers have fresh data
        if not (self.new_force and self.new_gt and self.new_position):
            return

        self.ax.cla()
        self.ax.set_ylim(-0.6, 4)
        self.ax.axis('equal')
        self.ax.axis('off')

        # Draw datasets with updated alpha
        for i, (xi, yi) in enumerate(self.datasets):
            self.ax.plot(xi, yi, color='black', linewidth=2)
            if i == self.Position1-1:
                alpha = float(np.clip(self.Force/500.0, 0.0, 1.0))
                self.ax.fill(xi, yi, color='red', alpha=alpha)

        self.ax.text(-0.65, 3.2, f"F={self.Force:.1f}g", fontsize=16, color='black')
        if self.propabilities[0]<0.7 and self.Force>0.1:
            circle = Circle(self.position, 0.2, fill=False, linewidth=3, color='black')
            self.ax.add_patch(circle)
        self.ax.scatter(self.x, self.y)

        # -------- RIGHT PLOT: F vs t --------
        self.ax2.cla()
        self.time_history.append(self.t)
        self.force_history.append(self.Force)
        self.gt_history.append(self.GT)
        self.t += 0.1  # loop step

        if len(self.time_history) > self.samples:
            self.time_history = self.time_history[-self.samples:]
            self.force_history = self.force_history[-self.samples:]
            self.gt_history = self.gt_history[-self.samples:]

        self.ax2.plot(self.time_history, self.force_history, linewidth=2)
        self.ax2.plot(self.time_history, self.gt_history, linewidth=2, linestyle="--", label="Ground Truth")
        self.ax2.set_title("Force vs Time")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Force (g)")
        self.ax2.grid(True)

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

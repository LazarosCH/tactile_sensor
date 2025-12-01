from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sensor_package',
            executable='Force_network_array',
            name='force_network'
        ),
        Node(
            package='sensor_package',
            executable='toutch_network',
            name='touch_network'
        ),
        Node(
            package='sensor_package',
            executable='vizualization_array',
            name='vizualization'
        ),
        TimerAction(
            period=2.0,  # seconds
            actions=[
                Node(
                    package='sensor_package',
                    executable='publisher',
                    name='publisher'
                ),
                Node(
                    package='sensor_package',
                    executable='groundT_publisher',
                    name='GT_publisher'
                )
            ]
        )
    ])
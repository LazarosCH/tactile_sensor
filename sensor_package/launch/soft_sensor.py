from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sensor_package',
            executable='force_network',
            name='force_network'
        ),
        Node(
            package='sensor_package',
            executable='position_network',
            name='position_network'
        ),
        Node(
            package='sensor_package',
            executable='vizualization',
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
                # Node(
                #     package='sensor_package',
                #     executable='groundT_publisher',
                #     name='publisher'
                # )
            ]
        )
    ])
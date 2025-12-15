from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable
from launch.substitutions import  PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    plotjuggler_config_file = PathJoinSubstitution(
        [FindPackageShare('sensor_package'), "config", "plotjuggler.xml"]
    )   

    return LaunchDescription([
        SetEnvironmentVariable( # important, since we need to find the other packages
            name='CUDA_VISIBLE_DEVICES',
            value='-1'
        ), 
        Node(
            package='sensor_package',
            executable='publisher',
            name='publisher'
        ),
        Node(
            package='sensor_package',
            executable='force_network',
            name='force_network'
        ),
        Node(
            package='sensor_package',
            executable='vizualization',
            name='vizualization'
        ),
        # Node(
        # package="plotjuggler",
        # executable="plotjuggler",
        # output="screen",
        # emulate_tty=True,
        # arguments=['-l', plotjuggler_config_file]
        # ),


        # TimerAction(
        #     period=2.0,  # seconds
        #     actions=[
        #         Node(
        #             package='sensor_package',
        #             executable='publisher',
        #             name='publisher'
        #         ),
                # Node(
                #     package='sensor_package',
                #     executable='groundT_publisher',
                #     name='publisher'
                # )
            # ]
        # )
    ])
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sensor_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=[],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='ROS2 node with TensorFlow',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'publisher = sensor_package.publisher:main',
            'force_network = sensor_package.Force_network:main',
            'position_network = sensor_package.position_network:main',
            'vizualization = sensor_package.vizualization:main',
            'Force_network_array = sensor_package.Force_network_array:main',
            'vizualization_array = sensor_package.vizualization_array:main',
            'groundT_publisher = sensor_package.publisher_groundTruth:main',
            'toutch_network = sensor_package.touch_network:main',
        ],
    },
)

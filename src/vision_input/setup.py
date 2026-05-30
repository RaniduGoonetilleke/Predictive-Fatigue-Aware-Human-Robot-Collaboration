from setuptools import find_packages, setup
import os  
from glob import glob 

package_name = 'vision_input'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ranidu P. Goonetilleke',
    maintainer_email='ranidugoonetilleke@gmail.com',
    description='Perception, fatigue monitoring and robot control nodes for a predictive, fatigue-aware human-robot collaboration system on the Universal Robots UR3 with ROS 2 Humble.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = vision_input.camera_node:main',
            'data_recorder = vision_input.data_recorder:main',
            'robot_controller = vision_input.robot_controller:main',
            'ar_pointing_interface = vision_input.ar_pointing_interface:main',
            'ar_robot_controller = vision_input.ar_robot_controller:main',
            'tobii_node = vision_input.tobii_node:main',
            'fatigue_monitor = vision_input.fatigue_monitor:main',
            'virtual_operator = vision_input.virtual_operator_node:main',
            'gamepad_operator = vision_input.gamepad_operator_node:main',
            'gamepad_sandbox = vision_input.gamepad_sandbox:main',
            'fatigue_override = vision_input.fatigue_override_node:main',
            'hand_mirror = vision_input.hand_mirror_node:main',
            'table_calibrator = vision_input.table_calibrator:main',
        ],
    },
)
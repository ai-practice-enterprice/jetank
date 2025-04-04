from setuptools import find_packages, setup

package_name = 'yolo_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, 
            ['package.xml']),
        ('share/' + package_name + '/weights',
            ['yolo_node/weights/best.pt']),
        ('share/' + package_name + '/classes',
            ['yolo_node/classes/classes.txt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        # executable_name = pkg_name.file_name:func_name 
        'console_scripts': [
            "yolo = yolo_node.yolov8_ros2_pt:main"
        ],
    },
)

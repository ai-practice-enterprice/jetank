from setuptools import find_packages, setup

package_name = 'jetank_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "navigate = jetank_navigation.jetank_FSMNavigator_node:main",
            "server = jetank_navigation.server_listener_node:main",
            "client = jetank_navigation.test_jetank_to_server:main",
        ],
    },
)

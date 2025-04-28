from setuptools import find_packages, setup

package_name = 'midas_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/' + package_name, [package_name+'/MidasDetector.py']),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'numpy',
        'opencv',
        'timm'
    ],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # executable_name = pkg_name.file_name:func_name 
            "depthV2 = midas_node.detectionV2:main"
        ],
    },
)

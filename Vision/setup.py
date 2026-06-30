import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'cobot3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),  # 모델 파일 등록
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='taehwan',
    maintainer_email='jolviadr@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'my_talker = cobot3.talker:main',
            'my_listener = cobot3.listener:main',
            'qr_decoder_node = cobot3.qr_decoder_node:main',
            'parcel_detector_node = cobot3.parcel_detector_node:main',
            'patchcore_anomaly_node = cobot3.patchcore_anomaly_node:main',
            'parcel_hub_node = cobot3.parcel_hub_node:main',
            'parcel_control_gui = cobot3.parcel_control_gui:main',
        ],
    },
)

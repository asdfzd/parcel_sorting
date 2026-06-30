"""
parcel_detector.launch.py
cobot3 패키지의 launch 디렉토리에 위치.
사용법: ros2 launch cobot3 parcel_detector.launch.py
"""
import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    default_model_path = os.path.join(
        get_package_share_directory('cobot3'),
        'models',
        'parcel_qr_det.pt'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_model_path,
        description='YOLO 모델 가중치 경로 (.pt 파일)'
    )
    rgb_topic_arg = DeclareLaunchArgument(
        'rgb_topic',
        default_value='/rgb/compressed',
        description='카메라 이미지 토픽'
    )
    conf_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='디텍션 신뢰도 임계값'
    )
    min_box_size_x_arg = DeclareLaunchArgument(
        'min_box_size_x',
        default_value='200',
        description='박스 전체 인식 최소 너비 (픽셀)'
    )
    min_box_size_y_arg = DeclareLaunchArgument(
        'min_box_size_y',
        default_value='150',
        description='박스 전체 인식 최소 높이 (픽셀)'
    )
    parcel_detector_node = Node(
        package='cobot3',
        executable='parcel_detector_node',
        name='parcel_detector_node',
        output='screen',
        parameters=[{
            'model_path':           LaunchConfiguration('model_path'),
            'rgb_topic':            LaunchConfiguration('rgb_topic'),
            'detection_topic':      '/parcel_detections',
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'target_class_ids':     [0, 1],
            'publish_annotated_image': True,
            'min_box_size_x':       LaunchConfiguration('min_box_size_x'),
            'min_box_size_y':       LaunchConfiguration('min_box_size_y'),
        }],
    )
    return LaunchDescription([
        model_path_arg,
        rgb_topic_arg,
        conf_threshold_arg,
        min_box_size_x_arg,
        min_box_size_y_arg,
        parcel_detector_node,
    ])

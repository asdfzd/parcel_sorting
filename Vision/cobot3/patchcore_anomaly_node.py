#!/usr/bin/env python3
"""
patchcore_anomaly_node.py

parcel_detector_node가 퍼블리시하는 /parcel_detections를 구독하고,
bbox로 /rgb 이미지를 crop한 뒤 PatchCore로 정상/훼손을 판별합니다.

결과를 /parcel_anomaly (std_msgs/String) 토픽으로 퍼블리시합니다.
  → "NORMAL"   : 정상 박스
  → "DAMAGED"  : 훼손 박스

cobot3 패키지에 통합:

"""

import os
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
from cv_bridge import CvBridge

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image as PILImage
import numpy as np
import cv2


class FeatureExtractor(torch.nn.Module):
    """patchcore_train.py와 동일한 구조 (반드시 일치해야 함)"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.layer2 = torch.nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        )
        self.layer3 = torch.nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )

    def forward(self, x):
        f2 = self.layer2(x)
        f3 = self.layer3(x)
        f3_up = torch.nn.functional.interpolate(
            f3, size=f2.shape[-2:], mode='bilinear', align_corners=False
        )
        return torch.cat([f2, f3_up], dim=1)


class PatchCoreAnomalyNode(Node):
    def __init__(self):
        super().__init__('patchcore_anomaly_node')

        # ---------------------------------------------------------
        # 파라미터
        # ---------------------------------------------------------
        _share = get_package_share_directory('cobot3')

        self.declare_parameter(
            'memory_bank_path',
            os.path.join(_share, 'models', 'patchcore_memory_bank.pt')
        )
        self.declare_parameter(
            'threshold_path',
            os.path.join(_share, 'models', 'patchcore_threshold.pt')
        )
        self.declare_parameter('rgb_topic', '/rgb')
        self.declare_parameter('detection_topic', '/parcel_detections')
        self.declare_parameter('result_topic', '/parcel_anomaly')
        self.declare_parameter('crop_padding', 10)
        # threshold_multiplier: 임계값을 몇 배로 적용할지 (1.0 = 그대로)
        # 오탐이 많으면 1.2~1.5로 높여서 민감도 낮출 수 있음
        self.declare_parameter('threshold_multiplier', 1.0)

        bank_path = self.get_parameter('memory_bank_path').get_parameter_value().string_value
        thresh_path = self.get_parameter('threshold_path').get_parameter_value().string_value
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.det_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.result_topic = self.get_parameter('result_topic').get_parameter_value().string_value
        self.pad = self.get_parameter('crop_padding').get_parameter_value().integer_value
        multiplier = self.get_parameter('threshold_multiplier').get_parameter_value().double_value

        # ---------------------------------------------------------
        # PatchCore 로드
        # ---------------------------------------------------------
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'디바이스: {self.device}')

        self.get_logger().info(f'메모리 뱅크 로딩: {bank_path}')
        self.memory_bank = torch.load(bank_path, map_location=self.device)
        self.threshold = float(torch.load(thresh_path)) * multiplier
        self.get_logger().info(f'임계값: {self.threshold:.4f} (multiplier: {multiplier})')

        self.extractor = FeatureExtractor().to(self.device)
        self.extractor.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # ---------------------------------------------------------
        # ROS2
        # ---------------------------------------------------------
        self.bridge = CvBridge()
        self.latest_image = None  # /rgb 최신 프레임 캐시

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.rgb_sub = self.create_subscription(
            Image, self.rgb_topic, self.rgb_callback, qos
        )
        self.det_sub = self.create_subscription(
            Detection2DArray, self.det_topic, self.detection_callback, 10
        )
        self.result_pub = self.create_publisher(String, self.result_topic, 10)

        self.get_logger().info(
            f'patchcore_anomaly_node 시작됨. '
            f'구독: {self.rgb_topic}, {self.det_topic} -> 퍼블리시: {self.result_topic}'
        )

    def rgb_callback(self, msg: Image):
        """최신 RGB 이미지를 캐시 (detection_callback에서 사용)"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'이미지 변환 실패: {e}')

    def detection_callback(self, msg: Detection2DArray):
        """YOLO 디텍션 결과마다 각 박스를 crop해서 이상 탐지"""
        if self.latest_image is None:
            return
        if not msg.detections:
            return

        img = self.latest_image
        h, w = img.shape[:2]
        results = []

        for det in msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            bw = det.bbox.size_x
            bh = det.bbox.size_y

            # center+size → xyxy 변환
            x1 = int(max(0, cx - bw / 2 - self.pad))
            y1 = int(max(0, cy - bh / 2 - self.pad))
            x2 = int(min(w, cx + bw / 2 + self.pad))
            y2 = int(min(h, cy + bh / 2 + self.pad))

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            score = self.compute_anomaly_score(crop)
            label = 'NORMAL' if score <= self.threshold else 'DAMAGED'

            cls_id = det.results[0].hypothesis.class_id if det.results else 'unknown'
            conf = det.results[0].hypothesis.score if det.results else 0.0

            results.append({
                'class': cls_id,
                'confidence': round(conf, 3),
                'anomaly_score': round(score, 4),
                'threshold': round(self.threshold, 4),
                'label': label,
                'bbox': [x1, y1, x2, y2],
            })

            self.get_logger().info(
                f'[{label}] score={score:.4f} (threshold={self.threshold:.4f}), '
                f'conf={conf:.3f}, bbox=[{x1},{y1},{x2},{y2}]'
            )

        if results:
            msg_out = String()
            msg_out.data = json.dumps(results, ensure_ascii=False)
            self.result_pub.publish(msg_out)

    def compute_anomaly_score(self, bgr_crop: np.ndarray) -> float:
        """crop된 BGR 이미지 → 이상 점수 계산"""
        rgb_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb_crop)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.extractor(tensor)
            B, C, H, W = feat.shape
            patches = feat.permute(0, 2, 3, 1).reshape(-1, C)
            dists = torch.cdist(patches, self.memory_bank)
            min_dists, _ = dists.min(dim=1)
            score = min_dists.max().item()

        return score


def main(args=None):
    rclpy.init(args=args)
    node = PatchCoreAnomalyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

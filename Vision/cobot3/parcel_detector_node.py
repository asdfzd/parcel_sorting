#!/usr/bin/env python3
"""
parcel_detector_node.py  (중앙 허브 연동 버전)

변경사항:
  - /hub/rgb/compressed 구독 (중앙 허브에서 재배포된 영상)
  - /state/detector 퍼블리시 (heartbeat + 감지 통계)
  - /cmd/detection_enable, /cmd/conf_threshold 구독 (허브 명령 수신)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Bool, Float32
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import cv2
import time
import json
from ultralytics import YOLO


class ParcelDetectorNode(Node):
    def __init__(self):
        super().__init__('parcel_detector_node')

        # ---------------------------------------------------------
        # 파라미터 선언 (런치 파일이나 CLI에서 오버라이드 가능)
        # ---------------------------------------------------------
        self.declare_parameter('model_path', 'yolo11n.pt')
        self.declare_parameter('rgb_topic', '/hub/rgb/compressed')  # 허브 재배포 토픽
        self.declare_parameter('detection_topic', '/parcel_detections')
        self.declare_parameter('confidence_threshold', 0.5)
        # COCO 기본 모델에는 '택배 박스' 클래스가 없으므로,
        # 커스텀 학습 전까지 임시로 사용할 클래스 ID 목록.
        # 커스텀 .pt로 교체 시 이 리스트를 새 클래스 ID로 바꿔주면 됨.
        self.declare_parameter('target_class_ids', [28])  # 28 = suitcase (임시 매핑)
        self.declare_parameter('publish_annotated_image', True)
        # 박스 전체가 잡혔는지 판단하는 최소 크기 (픽셀)
        self.declare_parameter('min_box_size_x', 200)
        self.declare_parameter('min_box_size_y', 180)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.target_class_ids = set(
            self.get_parameter('target_class_ids').get_parameter_value().integer_array_value
        )
        self.publish_annotated = self.get_parameter('publish_annotated_image').get_parameter_value().bool_value
        self.min_box_size_x = self.get_parameter('min_box_size_x').get_parameter_value().integer_value
        self.min_box_size_y = self.get_parameter('min_box_size_y').get_parameter_value().integer_value

        # ---------------------------------------------------------
        # 상태 통계 변수
        # ---------------------------------------------------------
        self._detect_count   = 0   # 총 감지 수
        self._frame_count    = 0   # 수신 프레임 수
        self._last_frame_t   = None
        self._fps_window     = []  # 최근 프레임 간격
        self._detect_enabled = True

        # ---------------------------------------------------------
        # YOLO 모델 로드
        # ---------------------------------------------------------
        self.get_logger().info(f'YOLO 모델 로딩 중: {model_path}')
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            self.get_logger().error(f'YOLO 모델 로드 실패: {e}')
            raise

        self.get_logger().info(f'모델 클래스 수: {len(self.model.names)}')
        self.get_logger().info(f'타겟 클래스 ID: {self.target_class_ids}')
        if not self.target_class_ids:
            self.get_logger().warn(
                '타겟 클래스 ID가 비어있습니다. 모든 클래스가 디텍션됩니다. '
                '커스텀 모델 학습 전까지는 의도된 설정일 수 있습니다.'
            )

        self.bridge = CvBridge()

        # ---------------------------------------------------------
        # QoS 설정: 센서 데이터는 BEST_EFFORT + 작은 큐가 일반적
        # (Isaac Sim 카메라 토픽과의 호환성 고려)
        # ---------------------------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = self.create_subscription(
            CompressedImage, self.rgb_topic, self.image_callback, qos
        )
        self.detection_pub = self.create_publisher(
            Detection2DArray, self.detection_topic, 10
        )

        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(
                Image, f'{self.detection_topic}/annotated', 10
            )

        # package + qr_label 쌍 확인 토픽 (qr_decoder_node 트리거용)
        self.paired_pub = self.create_publisher(
            Detection2DArray, '/parcel_with_qr', 10
        )

        # 송장 미부착 → /parcel_no_label 토픽으로 발행
        self.no_label_pub = self.create_publisher(
            String, '/parcel_no_label', 10
        )

        # 송장 미부착 박스 위치 추적 (중복 발행 방지)
        self._no_label_tracked  = {}
        self._no_label_next_id  = 0
        self._no_label_thresh   = 150

        # ── 상태 퍼블리셔 (/state/detector) ──────────────────────
        self.state_pub = self.create_publisher(String, '/state/detector', 10)
        # 1초마다 heartbeat 퍼블리시
        self.create_timer(1.0, self._publish_state)

        # ── 명령 구독자 (허브 → 이 노드) ──────────────────────
        self.create_subscription(Bool,    '/cmd/detection_enable',  self._cb_det_enable,  10)
        self.create_subscription(Float32, '/cmd/conf_threshold',    self._cb_conf,         10)

        self.get_logger().info(
            f'parcel_detector_node 시작됨. 구독: {self.rgb_topic} -> 퍼블리시: {self.detection_topic}'
        )

    def image_callback(self, msg: CompressedImage):
        if not self._detect_enabled:
            return
        # 프레임 수신 통계
        now = time.time()
        if self._last_frame_t is not None:
            interval = now - self._last_frame_t
            if 0 < interval < 5:
                self._fps_window.append(interval)
                if len(self._fps_window) > 30:
                    self._fps_window.pop(0)
        self._last_frame_t = now
        self._frame_count += 1

        try:
            import numpy as np
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            cv_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return
        except Exception as e:
            self.get_logger().error(f'이미지 변환 실패: {e}')
            return

        # YOLO 추론 (verbose=False로 콘솔 스팸 방지)
        results = self.model(cv_image, conf=self.conf_threshold, verbose=False)
        result = results[0]

        detection_array = Detection2DArray()
        detection_array.header = msg.header  # 원본 타임스탬프/프레임 유지 (시간 동기화에 중요)

        detected_count = 0
        matched_indices = []  # 필터링 통과한 박스의 원본 인덱스 (시각화 일치용)

        for idx, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])

            # 타겟 클래스 필터링 (빈 set이면 전부 통과)
            if self.target_class_ids and cls_id not in self.target_class_ids:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            det = Detection2D()
            det.header = msg.header

            # Detection2D는 center+size 형식 (xyxy가 아님에 주의)
            det.bbox.center.position.x = (x1 + x2) / 2.0
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = self.model.names.get(cls_id, str(cls_id))
            hypothesis.hypothesis.score = conf
            det.results.append(hypothesis)

            detection_array.detections.append(det)
            matched_indices.append(idx)
            detected_count += 1

        self.detection_pub.publish(detection_array)

        if detected_count > 0:
            self._detect_count += detected_count
            self.get_logger().debug(f'택배물 {detected_count}개 디텍션됨')

        # package + qr_label 쌍 매칭 후 /parcel_with_qr 발행
        self._publish_paired(detection_array, msg)

        # 시각화 이미지 퍼블리시 (RViz 디버깅용)
        # 필터링을 통과한 디텍션만 그려서 /parcel_detections 결과와 화면이 항상 일치하도록 함
        if self.publish_annotated:
            filtered_result = result[matched_indices] if matched_indices else result[[]]
            annotated = filtered_result.plot()
            try:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)
            except Exception as e:
                self.get_logger().error(f'시각화 이미지 퍼블리시 실패: {e}')


    # ── 상태 퍼블리시 ─────────────────────────────────────────
    def _publish_state(self):
        fps = 0.0
        if self._fps_window:
            avg = sum(self._fps_window) / len(self._fps_window)
            fps = round(1.0 / avg, 1) if avg > 0 else 0.0
        state = {
            "node":          "parcel_detector",
            "status":        "ok" if self._detect_enabled else "disabled",
            "fps":           fps,
            "detect_count":  self._detect_count,
            "frame_count":   self._frame_count,
            "conf":          self.conf_threshold,
            "enabled":       self._detect_enabled,
            "ts":            time.time(),
        }
        msg = String(); msg.data = json.dumps(state)
        self.state_pub.publish(msg)

    # ── 명령 콜백 ─────────────────────────────────────────────
    def _cb_det_enable(self, msg: Bool):
        self._detect_enabled = msg.data
        self.get_logger().info(f'감지 {"활성화" if msg.data else "비활성화"} 명령 수신')

    def _cb_conf(self, msg: Float32):
        self.conf_threshold = float(msg.data)
        self.get_logger().info(f'Conf threshold 변경: {self.conf_threshold:.2f}')

    def _publish_paired(self, detection_array, msg):
        """
        package + qr_label 쌍 → /parcel_with_qr 발행
        package만 있고 qr_label 없음 → /parcel_no_label 발행 (송장 미부착)
        """
        pkg_dets = [d for d in detection_array.detections
                    if d.results and d.results[0].hypothesis.class_id == 'package']
        qr_dets  = [d for d in detection_array.detections
                    if d.results and d.results[0].hypothesis.class_id == 'qr_label']

        if not pkg_dets:
            return

        # 박스 전체가 잡힌 것만 필터링 (끝부분만 감지된 작은 bbox 제외)
        pkg_dets = [d for d in pkg_dets
                    if d.bbox.size_x >= self.min_box_size_x
                    and d.bbox.size_y >= self.min_box_size_y]

        if not pkg_dets:
            self.get_logger().debug('박스 크기 미달 → 무시 (끝부분만 감지)')
            return

        if qr_dets:
            # package + qr_label 쌍 매칭 → /parcel_with_qr
            paired_array = Detection2DArray()
            paired_array.header = msg.header

            for pkg in pkg_dets:
                pcx = pkg.bbox.center.position.x
                pcy = pkg.bbox.center.position.y
                nearest_qr = min(qr_dets,
                    key=lambda q: abs(q.bbox.center.position.x - pcx)
                                + abs(q.bbox.center.position.y - pcy))
                paired_array.detections.append(pkg)
                paired_array.detections.append(nearest_qr)

            self.paired_pub.publish(paired_array)
            self.get_logger().debug(f'parcel_with_qr 발행: {len(pkg_dets)}개')

        else:
            # qr_label 없음 → 새 박스일 때만 NO_LABEL 발행
            current_cxcy = [(d.bbox.center.position.x, d.bbox.center.position.y) for d in pkg_dets]

            for pkg in pkg_dets:
                cx = pkg.bbox.center.position.x
                cy = pkg.bbox.center.position.y

                # 이미 추적 중인 박스인지 확인
                found_id = None
                for bid, info in self._no_label_tracked.items():
                    if abs(info['last_cx'] - cx) + abs(info['last_cy'] - cy) < self._no_label_thresh:
                        found_id = bid
                        break

                if found_id is not None:
                    self._no_label_tracked[found_id]['last_cx'] = cx
                    self._no_label_tracked[found_id]['last_cy'] = cy
                else:
                    # 새 박스 → 발행
                    self._no_label_tracked[self._no_label_next_id] = {'last_cx': cx, 'last_cy': cy}
                    self._no_label_next_id += 1
                    out_msg = String()
                    out_msg.data = 'NO_LABEL'
                    self.no_label_pub.publish(out_msg)
                    self.get_logger().info('송장 미부착 박스 → NO_LABEL 발행')

            # 화면에서 사라진 박스 제거
            active_ids = set()
            for cx, cy in current_cxcy:
                for bid, info in self._no_label_tracked.items():
                    if abs(info['last_cx'] - cx) + abs(info['last_cy'] - cy) < self._no_label_thresh:
                        active_ids.add(bid)
                        break
            for bid in list(self._no_label_tracked.keys()):
                if bid not in active_ids:
                    del self._no_label_tracked[bid]


def main(args=None):
    rclpy.init(args=args)
    node = ParcelDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

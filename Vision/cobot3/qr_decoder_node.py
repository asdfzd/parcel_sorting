#!/usr/bin/env python3
"""
qr_decoder_node.py  (cobot3 패키지)

YOLO가 감지한 qr_label bbox를 바로 crop → pyzbar 디코딩

동작 방식:
  /rgb + /parcel_detections 구독
  → package bbox: 위치 추적 (컨베이어 순서 정렬)
  → qr_label bbox: 바로 crop → pyzbar 디코딩
  → /qr_code 발행 (JSON)
  → /qr_crop_image 발행 (RViz 확인용)

발행 형식:
  {"order":1, "total":2, "zone":"ZONE_A", "center":[cx,cy]}
  (디코딩 성공 시에만 발행, NO_QR은 발행 안 함)
"""

import cv2
import queue
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Bool
import time
import json
from vision_msgs.msg import Detection2DArray


class QrDecoderNode(Node):

    def __init__(self):
        super().__init__('qr_decoder_node')

        self.declare_parameter('rgb_topic',          '/hub/rgb/compressed')  # 허브 재배포 토픽
        self.declare_parameter('detection_topic',    '/parcel_with_qr')

        # 상태 통계
        self._qr_success = 0
        self._qr_fail    = 0
        self._enabled    = True
        self.declare_parameter('output_topic',       '/qr_code')
        self.declare_parameter('conveyor_direction', 'bottom_to_top')
        self.declare_parameter('same_box_threshold', 150)
        self.declare_parameter('qr_padding',         10)

        self.rgb_topic   = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.det_topic   = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.out_topic   = self.get_parameter('output_topic').get_parameter_value().string_value
        self.direction   = self.get_parameter('conveyor_direction').get_parameter_value().string_value
        self.same_thresh = self.get_parameter('same_box_threshold').get_parameter_value().integer_value
        self.qr_pad      = self.get_parameter('qr_padding').get_parameter_value().integer_value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # 이미지 캐시
        self._cached_image  = None
        self._cached_header = None
        self._img_lock      = threading.Lock()

        self.rgb_sub = self.create_subscription(
            CompressedImage, self.rgb_topic, self.rgb_callback, qos
        )
        self.det_sub = self.create_subscription(
            Detection2DArray, self.det_topic, self.det_callback, 10
        )

        self.pub      = self.create_publisher(String, self.out_topic, 10)
        self.crop_pub = self.create_publisher(Image, '/qr_crop_image', 10)

        # ── 상태 퍼블리셔 ──────────────────────────────────────
        self.state_pub = self.create_publisher(String, '/state/qr_decoder', 10)
        self.create_timer(1.0, self._publish_state)

        # ── 명령 구독자 ────────────────────────────────────────
        self.create_subscription(Bool, '/cmd/qr_enable', self._cb_enable, 10)

        # 박스 위치 추적
        self.tracked    = {}
        self.next_id    = 0
        self._track_lock = threading.Lock()

        # QR 처리 전용 스레드
        self._qr_queue  = queue.Queue(maxsize=10)
        self._qr_thread = threading.Thread(target=self._qr_worker, daemon=True)
        self._qr_thread.start()

        self.get_logger().info(
            f'QR Decoder 시작됨 (YOLO qr_label bbox 방식)\n'
            f'  구독: {self.rgb_topic} + {self.det_topic}\n'
            f'  발행: {self.out_topic}\n'
            f'  컨베이어 방향: {self.direction}\n'
            f'  동일 박스 임계값: {self.same_thresh}px'
        )

    # ------------------------------------------------------------------
    def rgb_callback(self, msg: CompressedImage):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return
            with self._img_lock:
                self._cached_image  = img
                self._cached_header = msg.header
        except Exception as e:
            self.get_logger().error(f'이미지 변환 실패: {e}')

    # ------------------------------------------------------------------
    def _sort_packages(self, detections):
        """컨베이어 진입 순서대로 package 정렬"""
        if self.direction == 'bottom_to_top':
            return sorted(detections, key=lambda d: d.bbox.center.position.y)
        elif self.direction == 'top_to_bottom':
            return sorted(detections, key=lambda d: d.bbox.center.position.y, reverse=True)
        elif self.direction == 'left_to_right':
            return sorted(detections, key=lambda d: d.bbox.center.position.x)
        elif self.direction == 'right_to_left':
            return sorted(detections, key=lambda d: d.bbox.center.position.x, reverse=True)
        return detections

    def _find_tracked(self, cx, cy):
        best_id, best_dist = None, float('inf')
        for box_id, info in self.tracked.items():
            dist = abs(info['last_cx'] - cx) + abs(info['last_cy'] - cy)
            if dist < self.same_thresh and dist < best_dist:
                best_id, best_dist = box_id, dist
        return best_id

    def _cleanup_tracked(self, current_cxcy_list):
        active_ids = set()
        for cx, cy in current_cxcy_list:
            bid = self._find_tracked(cx, cy)
            if bid is not None:
                active_ids.add(bid)
        for bid in list(self.tracked.keys()):
            if bid not in active_ids:
                del self.tracked[bid]

    def _find_nearest_qr(self, pkg_cx, pkg_cy, qr_dets):
        """package에 가장 가까운 qr_label bbox 반환"""
        best, best_dist = None, float('inf')
        for qr in qr_dets:
            qcx = qr.bbox.center.position.x
            qcy = qr.bbox.center.position.y
            dist = abs(qcx - pkg_cx) + abs(qcy - pkg_cy)
            if dist < best_dist:
                best, best_dist = qr, dist
        return best

    # ------------------------------------------------------------------
    def det_callback(self, det_msg: Detection2DArray):
        with self._img_lock:
            cv_image = self._cached_image
            header   = self._cached_header

        if cv_image is None or not det_msg.detections:
            with self._track_lock:
                self.tracked.clear()
            return

        h, w = cv_image.shape[:2]

        # package / qr_label 분리
        pkg_dets = []
        qr_dets  = []
        for det in det_msg.detections:
            cls = det.results[0].hypothesis.class_id if det.results else ''
            if cls == 'package':
                pkg_dets.append(det)
            elif cls == 'qr_label':
                qr_dets.append(det)

        if not pkg_dets:
            with self._track_lock:
                self.tracked.clear()
            return

        sorted_pkgs = self._sort_packages(pkg_dets)
        total = len(sorted_pkgs)

        with self._track_lock:
            current_cxcy = [
                (d.bbox.center.position.x, d.bbox.center.position.y)
                for d in sorted_pkgs
            ]
            self._cleanup_tracked(current_cxcy)

            for idx, pkg in enumerate(sorted_pkgs):
                order = idx + 1
                cx = pkg.bbox.center.position.x
                cy = pkg.bbox.center.position.y

                # 가장 가까운 QR 찾기
                nearest_qr = self._find_nearest_qr(cx, cy, qr_dets)

                # QR crop 매 프레임 퍼블리시 (RViz 실시간 확인)
                qr_crop = None
                if nearest_qr is not None:
                    qcx = nearest_qr.bbox.center.position.x
                    qcy = nearest_qr.bbox.center.position.y
                    qbw = nearest_qr.bbox.size_x
                    qbh = nearest_qr.bbox.size_y
                    x1 = int(max(0, qcx - qbw/2 - self.qr_pad))
                    y1 = int(max(0, qcy - qbh/2 - self.qr_pad))
                    x2 = int(min(w, qcx + qbw/2 + self.qr_pad))
                    y2 = int(min(h, qcy + qbh/2 + self.qr_pad))
                    qr_crop = cv_image[y1:y2, x1:x2].copy()
                    if qr_crop.size > 0:
                        try:
                            self.crop_pub.publish(
                                self._cv2_to_imgmsg(qr_crop, header)
                            )
                        except Exception:
                            pass

                # 위치 추적 확인
                cur_id = self._find_tracked(cx, cy)
                if cur_id is not None:
                    self.tracked[cur_id]['last_cx'] = cx
                    self.tracked[cur_id]['last_cy'] = cy
                    # 이미 성공한 박스 → 스킵
                    if self.tracked[cur_id].get('zone') is not None:
                        continue
                    # 큐에서 처리 중 → 중복 방지
                    if self.tracked[cur_id].get('pending'):
                        continue
                else:
                    # 새 박스 → tracked 등록 후 cur_id 확보
                    cur_id = self.next_id
                    self.tracked[cur_id] = {
                        'last_cx': cx, 'last_cy': cy,
                        'zone': None, 'pending': False
                    }
                    self.next_id += 1

                # 디코딩 시도 (새 박스 or 실패한 박스, pending=False 확인됨)
                if qr_crop is not None and qr_crop.size > 0:
                    try:
                        self._qr_queue.put_nowait({
                            'crop':   qr_crop,
                            'cx':     cx,
                            'cy':     cy,
                            'box_id': cur_id,
                            'order':  order,
                            'total':  total,
                            'header': header,
                        })
                        self.tracked[cur_id]['pending'] = True
                    except queue.Full:
                        self.get_logger().debug('QR 큐 가득 참 → 스킵')
                else:
                    # qr_label bbox 없음 → parcel_detector_node에서 이미 처리
                    self.get_logger().debug('qr_label bbox 없음 → 스킵')

    # ------------------------------------------------------------------
    def _publish_result(self, order, total, zone, cx, cy):
        # NO_QR은 발행하지 않음 (parcel_detector_node에서 이미 처리)
        if zone == 'NO_QR':
            self._qr_fail += 1
            return
        self._qr_success += 1
        out_msg = String()
        out_msg.data = zone
        self.pub.publish(out_msg)
        self.get_logger().info(f'QR 발행: {zone}')

    # ── 상태 퍼블리시 ─────────────────────────────────────────
    def _publish_state(self):
        total = self._qr_success + self._qr_fail
        rate  = round(self._qr_success / total * 100, 1) if total > 0 else 0.0
        state = {
            "node":        "qr_decoder",
            "status":      "ok" if self._enabled else "disabled",
            "qr_success":  self._qr_success,
            "qr_fail":     self._qr_fail,
            "success_rate": rate,
            "enabled":     self._enabled,
            "ts":          time.time(),
        }
        msg = String(); msg.data = json.dumps(state)
        self.state_pub.publish(msg)

    # ── 명령 콜백 ─────────────────────────────────────────────
    def _cb_enable(self, msg: Bool):
        self._enabled = msg.data
        self.get_logger().info(f'QR 디코딩 {"활성화" if msg.data else "비활성화"}')

    # ------------------------------------------------------------------
    def _qr_worker(self):
        while True:
            try:
                item = self._qr_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            crop   = item['crop']
            cx     = item['cx']
            cy     = item['cy']
            box_id = item.get('box_id')
            order  = item['order']
            total  = item['total']
            header = item['header']

            # crop 퍼블리시
            try:
                self.crop_pub.publish(self._cv2_to_imgmsg(crop, header))
            except Exception:
                pass

            # pyzbar 디코딩
            zone = self._decode_qr(crop)
            self._publish_result(order, total, zone, cx, cy)

            # box_id로 정확한 박스에 zone 저장 (이동해도 덮어씌워지지 않음)
            with self._track_lock:
                if box_id is not None and box_id in self.tracked:
                    if zone != 'NO_QR':
                        self.tracked[box_id]['zone'] = zone
                        self.get_logger().debug(
                            f'박스 {box_id} zone 저장: {zone}'
                        )
                    self.tracked[box_id]['pending'] = False

            self._qr_queue.task_done()

    # ------------------------------------------------------------------
    def _decode_qr(self, crop) -> str:
        try:
            from pyzbar import pyzbar as pyzbar_lib

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            big  = cv2.resize(gray, None, fx=2, fy=2,
                              interpolation=cv2.INTER_LINEAR)
            _, binary = cv2.threshold(big, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            for img in (big, binary, gray):
                results = pyzbar_lib.decode(img)
                texts = [r.data.decode('utf-8') for r in results if r.data]
                if texts:
                    return texts[0]
        except Exception:
            pass

        # 폴백: OpenCV
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            data, _, _ = cv2.QRCodeDetector().detectAndDecode(gray)
            if data:
                return data
        except Exception:
            pass

        return 'NO_QR'

    def _cv2_to_imgmsg(self, cv_img: np.ndarray, header) -> Image:
        msg = Image()
        msg.header   = header
        msg.height   = cv_img.shape[0]
        msg.width    = cv_img.shape[1]
        msg.encoding = 'bgr8'
        msg.step     = cv_img.shape[1] * 3
        msg.data     = cv_img.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = QrDecoderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
parcel_hub_node.py — 중앙 컨트롤러 허브 노드

역할:
  1. /rgb/compressed 수신 → /hub/rgb/compressed 재배포
     (각 노드가 허브를 통해 영상을 받음 → 허브가 전송 제어 가능)

  2. 각 노드 상태 피드백 수신 및 검증
     /state/detector  → parcel_detector_node 상태
     /state/qr_decoder → qr_decoder_node 상태
     /state/simulation → Isaac Sim 상태

  3. 이상 감지 시 재명령 또는 경고 퍼블리시
     /hub/alert → GUI로 경고 전송

  4. GUI 명령을 각 노드에 중계
     GUI → /cmd/* → 허브 → 각 노드

토픽 구조:
  [Isaac Sim] → /rgb/compressed
                      ↓
               [parcel_hub_node]  ← /state/detector, /state/qr_decoder
                      ↓                    ↑ 재명령
        /hub/rgb/compressed          /cmd/detection_enable
                      ↓                    /cmd/conf_threshold
         [parcel_detector] → /parcel_detections
         [qr_decoder_node] → /qr_code
                      ↓
               [Control GUI] ← /hub/alert (경고)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool, Float32

import json
import time
import threading


# ── 설정 상수 ─────────────────────────────────────────────────────
TIMEOUT_DETECTOR   = 5.0   # detector 피드백 타임아웃(초) → 재명령
TIMEOUT_QR         = 8.0   # qr_decoder 피드백 타임아웃(초)
TIMEOUT_SIMULATION = 5.0   # Isaac Sim 피드백 타임아웃(초)
MAX_RETRY          = 3     # 최대 재명령 횟수


class ParcelHubNode(Node):
    def __init__(self):
        super().__init__('parcel_hub_node')

        # ── 파라미터 ─────────────────────────────────────────────
        self.declare_parameter('input_topic',  '/rgb/compressed')
        self.declare_parameter('output_topic', '/hub/rgb/compressed')
        self.declare_parameter('enable_watchdog', True)

        self._input_topic  = self.get_parameter('input_topic').get_parameter_value().string_value
        self._output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self._watchdog_on  = self.get_parameter('enable_watchdog').get_parameter_value().bool_value

        # ── 상태 추적 ─────────────────────────────────────────────
        self._state = {
            'detector':   {'last_ts': None, 'data': {}, 'retry': 0},
            'qr_decoder': {'last_ts': None, 'data': {}, 'retry': 0},
            'simulation': {'last_ts': None, 'data': {}, 'retry': 0},
        }
        self._frame_count    = 0
        self._relay_enabled  = True   # False면 /hub/rgb/compressed 재배포 중단
        self._lock           = threading.Lock()

        # 현재 명령 상태 (피드백 검증용)
        self._cmd_state = {
            'detection_enable': True,
            'conf_threshold':   0.5,
            'qr_enable':        True,
        }

        # ── QoS ──────────────────────────────────────────────────
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── 구독 ─────────────────────────────────────────────────
        # 영상 입력
        self.create_subscription(
            CompressedImage, self._input_topic,
            self._cb_image, qos_sensor)

        # 노드 상태 피드백
        self.create_subscription(
            String, '/state/detector',
            self._cb_state_detector, 10)
        self.create_subscription(
            String, '/state/qr_decoder',
            self._cb_state_qr, 10)
        self.create_subscription(
            String, '/state/simulation',
            self._cb_state_sim, 10)

        # GUI → 허브 명령 수신 (허브가 각 노드에 중계)
        self.create_subscription(Bool,    '/cmd/detection_enable', self._cb_cmd_det,  10)
        self.create_subscription(Float32, '/cmd/conf_threshold',   self._cb_cmd_conf, 10)
        self.create_subscription(Bool,    '/cmd/qr_enable',        self._cb_cmd_qr,   10)
        self.create_subscription(Bool,    '/cmd/relay_enable',     self._cb_cmd_relay, 10)

        # ── 퍼블리셔 ─────────────────────────────────────────────
        # 영상 재배포
        self._hub_pub = self.create_publisher(
            CompressedImage, self._output_topic, 1)

        # 노드로 명령 중계
        self._det_en_pub  = self.create_publisher(Bool,    '/cmd/detection_enable', 10)
        self._conf_pub    = self.create_publisher(Float32, '/cmd/conf_threshold',   10)
        self._qr_en_pub   = self.create_publisher(Bool,    '/cmd/qr_enable',        10)

        # GUI로 허브 상태 및 경고 전송
        self._hub_state_pub = self.create_publisher(String, '/hub/state',  10)
        self._alert_pub     = self.create_publisher(String, '/hub/alert',  10)

        # ── 타이머 ───────────────────────────────────────────────
        # 1초마다 허브 상태 퍼블리시
        self.create_timer(1.0, self._publish_hub_state)
        # 2초마다 워치독 실행
        if self._watchdog_on:
            self.create_timer(2.0, self._watchdog)

        self.get_logger().info(
            f'[HUB] parcel_hub_node 시작\n'
            f'  입력: {self._input_topic}\n'
            f'  출력: {self._output_topic}\n'
            f'  워치독: {"ON" if self._watchdog_on else "OFF"}'
        )

    # ── 영상 재배포 ───────────────────────────────────────────────
    def _cb_image(self, msg: CompressedImage):
        with self._lock:
            if not self._relay_enabled:
                return
            self._frame_count += 1
        self._hub_pub.publish(msg)

    # ── 상태 피드백 콜백 ─────────────────────────────────────────
    def _cb_state_detector(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self._lock:
                self._state['detector']['last_ts'] = time.time()
                self._state['detector']['data']    = data
                self._state['detector']['retry']   = 0  # 피드백 수신 시 재시도 카운트 리셋
        except Exception as e:
            self.get_logger().warn(f'[HUB] detector 상태 파싱 실패: {e}')

    def _cb_state_qr(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self._lock:
                self._state['qr_decoder']['last_ts'] = time.time()
                self._state['qr_decoder']['data']    = data
                self._state['qr_decoder']['retry']   = 0
        except Exception as e:
            self.get_logger().warn(f'[HUB] qr_decoder 상태 파싱 실패: {e}')

    def _cb_state_sim(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self._lock:
                self._state['simulation']['last_ts'] = time.time()
                self._state['simulation']['data']    = data
                self._state['simulation']['retry']   = 0
        except Exception as e:
            self.get_logger().warn(f'[HUB] simulation 상태 파싱 실패: {e}')

    # ── 명령 중계 콜백 ───────────────────────────────────────────
    def _cb_cmd_det(self, msg: Bool):
        with self._lock:
            self._cmd_state['detection_enable'] = msg.data
        self._det_en_pub.publish(msg)
        self.get_logger().info(f'[HUB] 감지 명령 중계: {"ON" if msg.data else "OFF"}')

    def _cb_cmd_conf(self, msg: Float32):
        with self._lock:
            self._cmd_state['conf_threshold'] = float(msg.data)
        self._conf_pub.publish(msg)
        self.get_logger().info(f'[HUB] Conf 명령 중계: {msg.data:.2f}')

    def _cb_cmd_qr(self, msg: Bool):
        with self._lock:
            self._cmd_state['qr_enable'] = msg.data
        self._qr_en_pub.publish(msg)
        self.get_logger().info(f'[HUB] QR 명령 중계: {"ON" if msg.data else "OFF"}')

    def _cb_cmd_relay(self, msg: Bool):
        with self._lock:
            self._relay_enabled = msg.data
        self.get_logger().info(f'[HUB] 영상 재배포 {"ON" if msg.data else "OFF"}')

    # ── 워치독 ───────────────────────────────────────────────────
    def _watchdog(self):
        """2초마다 각 노드 상태를 확인하고 이상 시 재명령 또는 경고"""
        now = time.time()
        with self._lock:
            state_copy = {k: dict(v) for k, v in self._state.items()}
            cmd_copy   = dict(self._cmd_state)

        # ── detector 검증 ─────────────────────────────────────
        det = state_copy['detector']
        if det['last_ts'] is None:
            self._alert('WARN', 'parcel_detector', '아직 피드백 없음 — 노드 시작 대기 중')
        else:
            elapsed = now - det['last_ts']
            if elapsed > TIMEOUT_DETECTOR:
                if det['retry'] < MAX_RETRY:
                    self._retry_detector(cmd_copy)
                    with self._lock:
                        self._state['detector']['retry'] += 1
                else:
                    self._alert('ERR', 'parcel_detector',
                                f'피드백 {elapsed:.0f}s 없음 — 노드 응답 없음 (재시도 {MAX_RETRY}회 초과)')
            else:
                # 피드백은 있는데 명령 상태와 불일치 확인
                actual_enabled = det['data'].get('enabled', True)
                expected       = cmd_copy['detection_enable']
                if actual_enabled != expected:
                    self.get_logger().warn(
                        f'[HUB][WATCHDOG] detector 상태 불일치 '
                        f'(기대={expected}, 실제={actual_enabled}) → 재명령')
                    self._retry_detector(cmd_copy)

        # ── qr_decoder 검증 ───────────────────────────────────
        qr = state_copy['qr_decoder']
        if qr['last_ts'] is None:
            self._alert('WARN', 'qr_decoder', '아직 피드백 없음 — 노드 시작 대기 중')
        else:
            elapsed = now - qr['last_ts']
            if elapsed > TIMEOUT_QR:
                if qr['retry'] < MAX_RETRY:
                    self._retry_qr(cmd_copy)
                    with self._lock:
                        self._state['qr_decoder']['retry'] += 1
                else:
                    self._alert('ERR', 'qr_decoder',
                                f'피드백 {elapsed:.0f}s 없음 — 노드 응답 없음')
            else:
                actual_enabled = qr['data'].get('enabled', True)
                expected       = cmd_copy['qr_enable']
                if actual_enabled != expected:
                    self.get_logger().warn('[HUB][WATCHDOG] qr_decoder 상태 불일치 → 재명령')
                    self._retry_qr(cmd_copy)

    def _retry_detector(self, cmd):
        """detector에 마지막 명령 상태로 재명령"""
        self.get_logger().warn('[HUB][WATCHDOG] parcel_detector 재명령 시도')
        m_en = Bool();  m_en.data  = cmd['detection_enable']
        m_cf = Float32(); m_cf.data = cmd['conf_threshold']
        self._det_en_pub.publish(m_en)
        self._conf_pub.publish(m_cf)
        self._alert('WARN', 'parcel_detector', '타임아웃 → 재명령 발송')

    def _retry_qr(self, cmd):
        """qr_decoder에 재명령"""
        self.get_logger().warn('[HUB][WATCHDOG] qr_decoder 재명령 시도')
        m_en = Bool(); m_en.data = cmd['qr_enable']
        self._qr_en_pub.publish(m_en)
        self._alert('WARN', 'qr_decoder', '타임아웃 → 재명령 발송')

    def _alert(self, level: str, node: str, msg: str):
        """GUI로 경고 전송"""
        payload = json.dumps({
            'level': level,
            'node':  node,
            'msg':   msg,
            'ts':    time.time(),
        })
        alert_msg = String(); alert_msg.data = payload
        self._alert_pub.publish(alert_msg)
        log_fn = self.get_logger().warn if level == 'WARN' else self.get_logger().error
        log_fn(f'[HUB][{level}] {node}: {msg}')

    # ── 허브 상태 퍼블리시 ────────────────────────────────────────
    def _publish_hub_state(self):
        now = time.time()
        with self._lock:
            state_copy = {k: dict(v) for k, v in self._state.items()}
            frame_cnt  = self._frame_count
            relay_en   = self._relay_enabled
            cmd_copy   = dict(self._cmd_state)

        def node_status(key, timeout):
            s = state_copy[key]
            if s['last_ts'] is None:
                return 'waiting'
            elapsed = now - s['last_ts']
            return 'ok' if elapsed < timeout else 'timeout'

        hub_state = {
            'node':         'parcel_hub',
            'relay_enabled': relay_en,
            'frame_count':   frame_cnt,
            'ts':            now,
            'nodes': {
                'parcel_detector': {
                    'status':       node_status('detector', TIMEOUT_DETECTOR),
                    'retry':        state_copy['detector']['retry'],
                    'data':         state_copy['detector']['data'],
                },
                'qr_decoder': {
                    'status':       node_status('qr_decoder', TIMEOUT_QR),
                    'retry':        state_copy['qr_decoder']['retry'],
                    'data':         state_copy['qr_decoder']['data'],
                },
                'isaac_sim': {
                    'status':       node_status('simulation', TIMEOUT_SIMULATION),
                    'data':         state_copy['simulation']['data'],
                },
            },
            'cmd_state': cmd_copy,
        }
        msg = String(); msg.data = json.dumps(hub_state)
        self._hub_state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ParcelHubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

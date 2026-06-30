#!/usr/bin/env python3
"""
Parcel Sorting & Inspection System - Central Control GUI
Isaac Sim + ROS2 연동 중앙 제어 패널 (비전 컨트롤러 포함)
"""

import sys
import os

# ─── cv2 내장 Qt 플러그인 충돌 방지 ─────────────────────────────
import importlib.util, pathlib as _pl
_cv2_spec = importlib.util.find_spec("cv2")
if _cv2_spec and _cv2_spec.origin:
    _cv2_qt = str(_pl.Path(_cv2_spec.origin).parent / "qt" / "plugins")
    _old    = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    if _cv2_qt in _old:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = \
            ":".join(p for p in _old.split(":") if p != _cv2_qt)

import subprocess, threading, time, datetime, json
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QScrollArea, QFrame,
    QStatusBar, QSizePolicy, QMessageBox, QSlider, QCheckBox,
    QComboBox, QTextEdit, QSplitter, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont, QPainter

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "0")
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import rclpy
    from sensor_msgs.msg import CompressedImage, Image
    from std_msgs.msg import String, Bool, Float32, Int32
    from vision_msgs.msg import Detection2DArray
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# ─── 저장 경로 ───────────────────────────────────────────────────
SAVE_DIR     = Path.home() / "parcel_captures"
QR_SAVE_DIR  = SAVE_DIR / "qr_crops"
PKG_SAVE_DIR = SAVE_DIR / "parcels"
for d in (SAVE_DIR, QR_SAVE_DIR, PKG_SAVE_DIR):
    d.mkdir(exist_ok=True)

# ─── ROS2 토픽 ───────────────────────────────────────────────────
CONVEYOR_TOPIC    = "/rgb/compressed"
PARCEL_QR_TOPIC   = "/parcel_with_qr"
PARCEL_NOQ_TOPIC  = "/parcel_no_label"
QR_CODE_TOPIC     = "/qr_code"
ANNOTATED_TOPIC   = "/parcel_detections/annotated"
QR_CROP_TOPIC     = "/qr_crop_image"
E_STOP_TOPIC      = "/emergency_stop"
RESET_TOPIC       = "/system_reset"
SIM_CTRL_TOPIC    = "/simulation_control"
# 비전 제어 토픽
CONF_TOPIC        = "/yolo_conf_threshold"
DETECT_EN_TOPIC   = "/detection_enable"
CAM_SEL_TOPIC     = "/camera_select"
SAVE_EN_TOPIC     = "/save_enable"
FPS_TOPIC         = "/detection_fps"
DETECT_CNT_TOPIC  = "/detection_count"
# ── 허브 연동 토픽 ────────────────────────────────────────────
HUB_STATE_TOPIC  = "/hub/state"        # 허브 → GUI (노드 상태 종합)
HUB_ALERT_TOPIC  = "/hub/alert"        # 허브 → GUI (경고/재명령 알림)
HUB_RELAY_TOPIC  = "/cmd/relay_enable" # GUI → 허브 (영상 재배포 ON/OFF)

DOMAIN_ID = 103

# ─── 색상 팔레트 ─────────────────────────────────────────────────
C_BG       = "#0D1117"
C_PANEL    = "#161B22"
C_BORDER   = "#30363D"
C_ACCENT   = "#00D9FF"
C_GREEN    = "#3FB950"
C_RED      = "#F85149"
C_ORANGE   = "#E3B341"
C_PURPLE   = "#A371F7"
C_TEXT     = "#E6EDF3"
C_SUBTEXT  = "#8B949E"
C_BTN_DARK = "#21262D"


# ═══════════════════════════════════════════════════════════════
#  ROS2 스레드
# ═══════════════════════════════════════════════════════════════
class Ros2Thread(QThread):
    frame_received       = pyqtSignal(bytes)
    parcel_qr_received   = pyqtSignal(str)    # Detection2DArray → JSON str
    parcel_noq_received  = pyqtSignal(bytes)
    qr_zone_received     = pyqtSignal(str)
    fps_received         = pyqtSignal(float)
    det_count_received   = pyqtSignal(int)
    annotated_received   = pyqtSignal(int, int, str, bytes)
    qr_crop_received     = pyqtSignal(int, int, str, bytes)
    hub_state_received   = pyqtSignal(str)   # 허브 종합 상태 JSON
    hub_alert_received   = pyqtSignal(str)   # 허브 경고 JSON

    @staticmethod
    def _raw_to_bytes(msg):
        """sensor_msgs/Image → (height, width, encoding, raw_bytes) 튜플 반환용 헬퍼"""
        return msg.height, msg.width, msg.encoding, bytes(msg.data)

    def __init__(self):
        super().__init__()
        self._running = False
        self.node = None
        self._pubs = {}

    def run(self):
        if not ROS2_AVAILABLE:
            return
        os.environ["ROS_DOMAIN_ID"] = str(DOMAIN_ID)
        rclpy.init()
        self.node = rclpy.create_node("parcel_gui_node")
        n = self.node

        # 구독
        n.create_subscription(CompressedImage, CONVEYOR_TOPIC,
            lambda m: self.frame_received.emit(bytes(m.data)), 10)
        # /parcel_with_qr 은 Detection2DArray — bbox로 /rgb에서 크롭
        n.create_subscription(Detection2DArray, PARCEL_QR_TOPIC,
            lambda m: self.parcel_qr_received.emit(self._det_to_json(m)), 10)
        n.create_subscription(CompressedImage, PARCEL_NOQ_TOPIC,
            lambda m: self.parcel_noq_received.emit(bytes(m.data)), 10)
        n.create_subscription(String, QR_CODE_TOPIC,
            lambda m: self.qr_zone_received.emit(m.data), 10)
        n.create_subscription(Image, ANNOTATED_TOPIC,
            lambda m: self.annotated_received.emit(m.height, m.width, m.encoding, bytes(m.data)), 10)
        n.create_subscription(Image, QR_CROP_TOPIC,
            lambda m: self.qr_crop_received.emit(m.height, m.width, m.encoding, bytes(m.data)), 10)
        # 허브 상태 및 경고
        n.create_subscription(String, HUB_STATE_TOPIC,
            lambda m: self.hub_state_received.emit(m.data), 10)
        n.create_subscription(String, HUB_ALERT_TOPIC,
            lambda m: self.hub_alert_received.emit(m.data), 10)
        n.create_subscription(Float32, FPS_TOPIC,
            lambda m: self.fps_received.emit(float(m.data)), 10)
        n.create_subscription(Int32, DETECT_CNT_TOPIC,
            lambda m: self.det_count_received.emit(int(m.data)), 10)

        # 퍼블리셔
        self._pubs = {
            "estop":    n.create_publisher(Bool,    E_STOP_TOPIC,   1),
            "reset":    n.create_publisher(Bool,    RESET_TOPIC,    1),
            "conf":     n.create_publisher(Float32, CONF_TOPIC,     1),
            "det_en":   n.create_publisher(Bool,    DETECT_EN_TOPIC,1),
            "cam_sel":  n.create_publisher(String,  CAM_SEL_TOPIC,  1),
            "save_en":  n.create_publisher(Bool,    SAVE_EN_TOPIC,  1),
            "sim_ctrl": n.create_publisher(String,  SIM_CTRL_TOPIC, 1),
        }

        self._running = True
        while self._running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.node.destroy_node()
        rclpy.shutdown()

    def _pub_bool(self, key, val):
        if self.node and ROS2_AVAILABLE:
            m = Bool(); m.data = val
            self._pubs[key].publish(m)

    def _pub_float(self, key, val):
        if self.node and ROS2_AVAILABLE:
            m = Float32(); m.data = float(val)
            self._pubs[key].publish(m)

    def _pub_str(self, key, val):
        if self.node and ROS2_AVAILABLE:
            m = String(); m.data = val
            self._pubs[key].publish(m)

    def publish_e_stop(self, v):    self._pub_bool("estop",    v)
    def publish_reset(self):        self._pub_bool("reset",    True)
    def publish_conf(self, v):      self._pub_float("conf",    v)
    def publish_det_enable(self,v): self._pub_bool("det_en",   v)
    def publish_cam_select(self,v): self._pub_str("cam_sel",   v)
    def publish_save_enable(self,v):self._pub_bool("save_en",  v)
    def publish_sim_ctrl(self, cmd: str):
        self._pub_str("sim_ctrl", cmd)

    @staticmethod
    def _det_to_json(msg) -> str:
        """Detection2DArray → JSON 문자열 (bbox 리스트)"""
        import json as _json
        dets = []
        for d in msg.detections:
            cls = d.results[0].hypothesis.class_id if d.results else ""
            dets.append({
                "cls":  cls,
                "cx":   d.bbox.center.position.x,
                "cy":   d.bbox.center.position.y,
                "w":    d.bbox.size_x,
                "h":    d.bbox.size_y,
                "conf": d.results[0].hypothesis.score if d.results else 0.0,
            })
        return _json.dumps(dets)

    def stop(self):
        self._running = False
        self.wait(2000)


# ═══════════════════════════════════════════════════════════════
#  카메라 뷰 위젯
# ═══════════════════════════════════════════════════════════════
class CameraView(QLabel):
    def __init__(self, title=""):
        super().__init__()
        self._title = title
        self._has_frame = False
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_t = time.time()
        self.setMinimumSize(280, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background:#10171E; border:1px solid {C_BORDER}; border-radius:6px;")
        self._draw_placeholder()

    def _draw_placeholder(self):
        w, h = max(self.width(), 280), max(self.height(), 200)
        pm = QPixmap(w, h); pm.fill(QColor("#10171E"))
        p = QPainter(pm)
        p.setPen(QColor(C_BORDER))
        p.drawLine(0,0,w,h); p.drawLine(w,0,0,h)
        p.setPen(QColor(C_SUBTEXT))
        p.setFont(QFont("Consolas", 10))
        p.drawText(pm.rect(), Qt.AlignCenter, f"[ {self._title} ]\n신호 없음")
        p.end()
        self.setPixmap(pm)

    def update_from_compressed(self, data: bytes):
        if not CV2_AVAILABLE: return
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            self._set_cv(img)

    def update_from_cv(self, img):
        self._set_cv(img)

    def _set_cv(self, img):
        self._frame_count += 1
        now = time.time()
        if now - self._last_fps_t >= 1.0:
            self._fps = self._frame_count / (now - self._last_fps_t)
            self._frame_count = 0
            self._last_fps_t = now
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pm)
        self._has_frame = True

    def get_fps(self): return self._fps

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._has_frame: self._draw_placeholder()


# ═══════════════════════════════════════════════════════════════
#  상태 뱃지
# ═══════════════════════════════════════════════════════════════
class StatusBadge(QLabel):
    def __init__(self):
        super().__init__("● IDLE")
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(24)
        self.set_state("idle")

    def set_state(self, s):
        cfg = {
            "idle":    (C_BTN_DARK, C_SUBTEXT, "●  IDLE"),
            "running": (C_GREEN,    "#000",     "●  RUNNING"),
            "stopped": (C_RED,      "#fff",     "●  E-STOP"),
        }
        bg, fg, txt = cfg.get(s, cfg["idle"])
        self.setText(txt)
        self.setStyleSheet(f"""
            QLabel {{
                background:{bg}; color:{fg}; border-radius:12px;
                padding:0 12px; font-size:13px; font-weight:bold;
                font-family:Consolas;
            }}
        """)


# ═══════════════════════════════════════════════════════════════
#  로그 위젯
# ═══════════════════════════════════════════════════════════════
class LogWidget(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumHeight(130)
        self.setStyleSheet(f"""
            QTextEdit {{
                background:#0D1117; color:{C_SUBTEXT};
                border:1px solid {C_BORDER}; border-radius:6px;
                font-family:Consolas; font-size:13px;
                padding:4px;
            }}
        """)

    def log(self, msg: str, level="INFO"):
        colors = {"INFO": C_SUBTEXT, "OK": C_GREEN, "WARN": C_ORANGE,
                  "ERR": C_RED, "CMD": C_ACCENT, "VISION": C_PURPLE}
        col = colors.get(level, C_SUBTEXT)
        ts  = datetime.datetime.now().strftime("%H:%M:%S")
        self.append(
            f'<span style="color:{C_BORDER}">[{ts}]</span> '
            f'<span style="color:{col}">[{level}]</span> '
            f'<span style="color:{C_TEXT}">{msg}</span>'
        )
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


# ═══════════════════════════════════════════════════════════════
#  썸네일 위젯
# ═══════════════════════════════════════════════════════════════
class ThumbnailWidget(QWidget):
    def __init__(self, label=""):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setSpacing(3); lay.setContentsMargins(0,0,0,0)
        self.img = QLabel()
        self.img.setFixedSize(120, 86)
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setStyleSheet(f"background:#1C2128; border:1px solid {C_BORDER}; border-radius:4px; color:{C_SUBTEXT}; font-size:12px;")
        self.img.setText("없음")
        self.name = QLabel(label)
        self.name.setAlignment(Qt.AlignCenter)
        self.name.setStyleSheet(f"color:{C_SUBTEXT}; font-size:11px;")
        self.name.setWordWrap(True)
        lay.addWidget(self.img); lay.addWidget(self.name)

    def set_image(self, pm: QPixmap, fn: str):
        self.img.setPixmap(pm.scaled(120,86,Qt.KeepAspectRatio,Qt.SmoothTransformation))
        self.name.setText(fn[-22:])


# ═══════════════════════════════════════════════════════════════
#  메인 윈도우
# ═══════════════════════════════════════════════════════════════
class ParcelControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parcel Sorting & Inspection — Central Controller")
        self.setMinimumSize(1400, 860)

        self._state        = "idle"
        self._isaac_proc   = None
        self._parcel_count = 0
        self._qr_count     = 0
        self._zone_a_cnt   = 0
        self._zone_b_cnt   = 0
        self._no_label_cnt = 0   # 라벨 없는 택배 카운트
        self._det_enabled  = True
        self._save_enabled = True
        self._cam_top      = True
        self._conf_val     = 0.50

        # ── QR 베스트샷 버퍼 ─────────────────────────────────
        self._BESTSHOT_WINDOW = 1.5
        self._qr_bestshot_active  = False
        self._qr_bestshot_buf     = []
        self._qr_bestshot_zone    = ""
        self._qr_bestshot_timer   = QTimer()
        self._qr_bestshot_timer.setSingleShot(True)
        self._qr_bestshot_timer.timeout.connect(self._flush_qr_bestshot)

        # 택배 bestshot 버퍼
        self._pkg_bestshot_active = False
        self._pkg_bestshot_buf    = []
        self._pkg_bestshot_timer  = QTimer()
        self._pkg_bestshot_timer.setSingleShot(True)
        self._pkg_bestshot_timer.timeout.connect(self._flush_pkg_bestshot)

        # ── 토픽 상태 추적 ────────────────────────────────────
        self._topic_last_recv = {
            "/rgb/compressed":                  None,
            "/parcel_detections/annotated":     None,
            "/parcel_with_qr":                  None,
            "/parcel_no_label":                 None,
            "/qr_crop_image":                   None,
            "/qr_code":                         None,
        }
        self._topic_timeout = {
            "/rgb/compressed":                  2.0,
            "/parcel_detections/annotated":     3.0,
            "/parcel_with_qr":                  3.0,
            "/parcel_no_label":                 5.0,
            "/qr_crop_image":                   3.0,
            "/qr_code":                         5.0,
        }

        # ── 자동 진단용 측정 데이터 ──────────────────────────
        # 기대 Hz 범위 (min, max) — 이 범위 벗어나면 WARN
        self._topic_hz_range = {
            "/rgb/compressed":                  (20.0, 60.0),
            "/parcel_detections/annotated":     (15.0, 60.0),
            "/parcel_with_qr":                  (0.1,  30.0),
            "/parcel_no_label":                 (0.1,  30.0),
            "/qr_crop_image":                   (0.1,  30.0),
            "/qr_code":                         (0.1,  10.0),
        }
        # 수신 카운트 & 바이트 (1초 윈도우 측정)
        self._topic_recv_cnt  = {k: 0   for k in self._topic_last_recv}
        self._topic_recv_bw   = {k: 0   for k in self._topic_last_recv}  # bytes/s
        self._topic_hz_cur    = {k: 0.0 for k in self._topic_last_recv}
        self._topic_bw_cur    = {k: 0.0 for k in self._topic_last_recv}
        self._topic_hz_window = {k: []  for k in self._topic_last_recv}  # 최근 간격(초)
        # 자동 진단 이력 로그
        self._diag_log: list[tuple[str,str,str]] = []  # (시각, 레벨, 메시지)
        self._auto_diag_running = False

        self._ros = Ros2Thread()
        self._ros.frame_received.connect(self._on_conveyor)
        self._ros.parcel_qr_received.connect(self._on_parcel_qr)
        self._ros.parcel_noq_received.connect(self._on_parcel_noq)
        self._ros.qr_zone_received.connect(self._on_qr_zone)
        self._ros.fps_received.connect(self._on_fps)
        self._ros.det_count_received.connect(self._on_det_count)
        self._ros.annotated_received.connect(self._on_annotated)
        self._ros.qr_crop_received.connect(self._on_qr_crop)
        self._ros.hub_state_received.connect(self._on_hub_state)
        self._ros.hub_alert_received.connect(self._on_hub_alert)

        # 허브 노드 상태 캐시
        self._hub_state: dict = {}
        self._build_ui()
        self._apply_style()

        self._clock_timer = QTimer()
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()

        # FPS 표시 갱신 타이머
        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._refresh_fps_display)
        self._fps_timer.start(500)

        # 진단 타이머 (1초마다 토픽 상태 갱신)
        self._diag_timer = QTimer()
        self._diag_timer.timeout.connect(self._refresh_diag)
        self._diag_timer.start(1000)

        if not ROS2_AVAILABLE:
            self._demo_t = QTimer()
            self._demo_t.timeout.connect(self._demo_frame)
            self._demo_t.start(80)

        self._autostart_done = False  # showEvent에서 1회만 실행

    # ──────────────────────────────────────────────────────────
    #  UI 구성
    # ──────────────────────────────────────────────────────────
    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setSpacing(0); root.setContentsMargins(0,0,0,0)
        root.addWidget(self._mk_topbar())

        # 메인 바디: 왼쪽 패널 | 오른쪽(영상+캡처)
        body = QHBoxLayout()
        body.setContentsMargins(10,8,10,8)
        body.setSpacing(10)

        # 왼쪽: 탭으로 "기본 제어" / "비전 제어"
        body.addWidget(self._mk_left_panel(), stretch=0)

        # 오른쪽
        right = QVBoxLayout(); right.setSpacing(8)
        right.addWidget(self._mk_video_panel(), stretch=4)
        right.addWidget(self._mk_log_panel(),   stretch=1)
        right.addWidget(self._mk_capture_panel(), stretch=2)
        rw = QWidget(); rw.setLayout(right)
        body.addWidget(rw, stretch=1)

        bw = QWidget(); bw.setLayout(body)
        root.addWidget(bw, stretch=1)
        self._mk_statusbar()

    # ── 상단 바 ───────────────────────────────────────────────
    def _mk_topbar(self):
        bar = QWidget(); bar.setFixedHeight(50)
        bar.setStyleSheet(f"background:{C_PANEL}; border-bottom:1px solid {C_BORDER};")
        lay = QHBoxLayout(bar); lay.setContentsMargins(16,0,16,0)

        title = QLabel("⬡  PARCEL CENTRAL CONTROLLER")
        title.setStyleSheet(f"color:{C_ACCENT}; font-size:16px; font-weight:bold; font-family:Consolas; letter-spacing:2px;")
        lay.addWidget(title)

        self._badge = StatusBadge()
        lay.addWidget(self._badge)
        lay.addStretch()

        # 비전 상태 인디케이터
        self._lbl_det_indicator = QLabel("● 감지 ON")
        self._lbl_det_indicator.setStyleSheet(f"color:{C_GREEN}; font-family:Consolas; font-size:13px;")
        lay.addWidget(self._lbl_det_indicator)

        sep = QLabel("|"); sep.setStyleSheet(f"color:{C_BORDER}; margin:0 8px;")
        lay.addWidget(sep)

        self._lbl_cam_indicator = QLabel("CAM: 상단")
        self._lbl_cam_indicator.setStyleSheet(f"color:{C_ACCENT}; font-family:Consolas; font-size:13px;")
        lay.addWidget(self._lbl_cam_indicator)

        sep2 = QLabel("|"); sep2.setStyleSheet(f"color:{C_BORDER}; margin:0 8px;")
        lay.addWidget(sep2)

        ros_lbl = QLabel("ROS2 ✔" if ROS2_AVAILABLE else "ROS2 ✖ 데모")
        ros_lbl.setStyleSheet(f"color:{'#3FB950' if ROS2_AVAILABLE else C_ORANGE}; font-family:Consolas; font-size:13px;")
        lay.addWidget(ros_lbl)

        self._clock_lbl = QLabel()
        self._clock_lbl.setStyleSheet(f"color:{C_SUBTEXT}; font-family:Consolas; font-size:13px; margin-left:14px;")
        lay.addWidget(self._clock_lbl)
        return bar

    # ── 왼쪽 탭 패널 ──────────────────────────────────────────
    def _mk_left_panel(self):
        tabs = QTabWidget()
        tabs.setFixedWidth(280)
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background:{C_PANEL}; border:1px solid {C_BORDER}; border-radius:8px;
            }}
            QTabBar::tab {{
                background:{C_BTN_DARK}; color:{C_SUBTEXT};
                padding:6px 10px; font-family:Consolas; font-size:13px;
                border:1px solid {C_BORDER};
            }}
            QTabBar::tab:selected {{
                background:{C_PANEL}; color:{C_TEXT};
                border-bottom:2px solid {C_ACCENT};
            }}
        """)
        tabs.addTab(self._mk_system_tab(), "시스템")
        tabs.addTab(self._mk_vision_tab(), "비전")
        return tabs

    # ── 시스템 탭 ─────────────────────────────────────────────
    def _mk_system_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w); lay.setSpacing(8); lay.setContentsMargins(10,10,10,10)

        # ── 시뮬레이션 상태 박스 ──────────────────────────────
        sim_grp = self._mk_group("시뮬레이션 상태", inner=True)
        sim_lay = QVBoxLayout(sim_grp); sim_lay.setSpacing(8); sim_lay.setContentsMargins(8,8,8,8)

        self._lbl_sim_state = QLabel("● 대기 중")
        self._lbl_sim_state.setAlignment(Qt.AlignCenter)
        self._lbl_sim_state.setStyleSheet(
            f"color:{C_SUBTEXT}; font-family:Consolas; font-size:14px; font-weight:bold; padding:6px;")
        sim_lay.addWidget(self._lbl_sim_state)

        zone_cnt_row = QHBoxLayout(); zone_cnt_row.setSpacing(8)
        za_box = QWidget()
        za_box.setStyleSheet(f"background:{C_BTN_DARK}; border-radius:6px;")
        za_lay = QVBoxLayout(za_box); za_lay.setContentsMargins(6,4,6,4); za_lay.setSpacing(1)
        self._lbl_za_cnt = QLabel("0")
        self._lbl_za_cnt.setAlignment(Qt.AlignCenter)
        self._lbl_za_cnt.setStyleSheet(f"color:{C_GREEN}; font-family:Consolas; font-size:22px; font-weight:bold;")
        za_lbl = QLabel("A 완료")
        za_lbl.setAlignment(Qt.AlignCenter)
        za_lbl.setStyleSheet(f"color:{C_SUBTEXT}; font-family:Consolas; font-size:11px;")
        za_lay.addWidget(self._lbl_za_cnt); za_lay.addWidget(za_lbl)

        zb_box = QWidget()
        zb_box.setStyleSheet(f"background:{C_BTN_DARK}; border-radius:6px;")
        zb_lay = QVBoxLayout(zb_box); zb_lay.setContentsMargins(6,4,6,4); zb_lay.setSpacing(1)
        self._lbl_zb_cnt = QLabel("0")
        self._lbl_zb_cnt.setAlignment(Qt.AlignCenter)
        self._lbl_zb_cnt.setStyleSheet(f"color:{C_ACCENT}; font-family:Consolas; font-size:22px; font-weight:bold;")
        zb_lbl = QLabel("B 완료")
        zb_lbl.setAlignment(Qt.AlignCenter)
        zb_lbl.setStyleSheet(f"color:{C_SUBTEXT}; font-family:Consolas; font-size:11px;")
        zb_lay.addWidget(self._lbl_zb_cnt); zb_lay.addWidget(zb_lbl)

        zone_cnt_row.addWidget(za_box); zone_cnt_row.addWidget(zb_box)
        sim_lay.addLayout(zone_cnt_row)
        lay.addWidget(sim_grp)
        lay.addWidget(self._divider())

        self._btn_start = self._mk_btn("▶  시  작",   C_GREEN,  "#000", h=50)
        self._btn_estop = self._mk_btn("⬛  긴급 정지", C_RED,   "#fff", h=50)
        self._btn_reset = self._mk_btn("↺  초  기  화", C_ORANGE,"#000", h=40)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_estop.clicked.connect(self._on_estop)
        self._btn_reset.clicked.connect(self._on_reset)
        lay.addWidget(self._btn_start)
        lay.addWidget(self._btn_estop)
        lay.addWidget(self._btn_reset)
        lay.addWidget(self._divider())

        # 통계
        stats = self._mk_group("통계", inner=True)
        sl = QVBoxLayout(stats); sl.setSpacing(5)
        self._s_total = self._stat_row(sl, "총 택배",    "0")
        self._s_qr    = self._stat_row(sl, "QR 인식",   "0")
        self._s_za    = self._stat_row(sl, "ZONE A",    "0")
        self._s_zb    = self._stat_row(sl, "ZONE B",    "0")
        self._s_nl    = self._stat_row(sl, "라벨 없음", "0")
        lay.addWidget(stats)

        btn_f = self._mk_btn("📁  저장 폴더", C_BTN_DARK, C_SUBTEXT, h=30, small=True)
        btn_f.clicked.connect(lambda: subprocess.Popen(["xdg-open", str(SAVE_DIR)]))
        lay.addWidget(btn_f)

        btn_diag = self._mk_btn("📡  시스템 진단", C_ACCENT, "#000", h=30, small=True)
        btn_diag.clicked.connect(self._open_diag_window)
        lay.addWidget(btn_diag)

        btn_log = self._mk_btn("📋  이벤트 로그", C_BTN_DARK, C_TEXT, h=30, small=True)
        btn_log.clicked.connect(self._open_log_window)
        lay.addWidget(btn_log)

        btn_chart = self._mk_btn("📊  통계 그래프", C_GREEN, "#000", h=30, small=True)
        btn_chart.clicked.connect(self._open_chart_window)
        lay.addWidget(btn_chart)
        lay.addStretch()

        # 최근 QR
        qz = self._mk_group("최근 QR", inner=True)
        ql = QVBoxLayout(qz)
        self._lbl_zone = QLabel("—")
        self._lbl_zone.setAlignment(Qt.AlignCenter)
        self._lbl_zone.setStyleSheet(f"color:{C_ACCENT}; font-family:Consolas; font-size:20px; font-weight:bold; padding:8px;")
        ql.addWidget(self._lbl_zone)
        lay.addWidget(qz)
        return w

    # ── 비전 제어 탭 ──────────────────────────────────────────
    def _mk_vision_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w); lay.setSpacing(10); lay.setContentsMargins(10,10,10,10)

        # ── 감지 ON/OFF ───────────────────────────────────────
        det_grp = self._mk_group("감지 제어", inner=True)
        dg = QVBoxLayout(det_grp); dg.setSpacing(8)

        self._chk_detect = QCheckBox("YOLO 감지 활성화")
        self._chk_detect.setChecked(True)
        self._chk_detect.setStyleSheet(f"""
            QCheckBox {{ color:{C_TEXT}; font-family:Consolas; font-size:13px; }}
            QCheckBox::indicator {{ width:16px; height:16px; border:1px solid {C_BORDER}; border-radius:3px; background:{C_BTN_DARK}; }}
            QCheckBox::indicator:checked {{ background:{C_GREEN}; border-color:{C_GREEN}; }}
        """)
        self._chk_detect.stateChanged.connect(self._on_det_toggle)
        dg.addWidget(self._chk_detect)

        self._chk_save = QCheckBox("이미지 자동 저장")
        self._chk_save.setChecked(True)
        self._chk_save.setStyleSheet(self._chk_detect.styleSheet())
        self._chk_save.stateChanged.connect(self._on_save_toggle)
        dg.addWidget(self._chk_save)

        lay.addWidget(det_grp)

        # ── Confidence Threshold ──────────────────────────────
        conf_grp = self._mk_group("Confidence Threshold", inner=True)
        cg = QVBoxLayout(conf_grp); cg.setSpacing(6)

        conf_hdr = QHBoxLayout()
        conf_lbl = QLabel("임계값")
        conf_lbl.setStyleSheet(f"color:{C_SUBTEXT}; font-size:13px;")
        self._lbl_conf_val = QLabel("0.50")
        self._lbl_conf_val.setStyleSheet(f"color:{C_ACCENT}; font-family:Consolas; font-size:16px; font-weight:bold;")
        conf_hdr.addWidget(conf_lbl); conf_hdr.addStretch(); conf_hdr.addWidget(self._lbl_conf_val)
        cg.addLayout(conf_hdr)

        self._slider_conf = QSlider(Qt.Horizontal)
        self._slider_conf.setRange(10, 95)
        self._slider_conf.setValue(50)
        self._slider_conf.setTickInterval(10)
        self._slider_conf.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background:{C_BTN_DARK}; height:4px; border-radius:2px;
            }}
            QSlider::handle:horizontal {{
                background:{C_ACCENT}; width:14px; height:14px;
                border-radius:7px; margin:-5px 0;
            }}
            QSlider::sub-page:horizontal {{
                background:{C_ACCENT}; border-radius:2px;
            }}
        """)
        self._slider_conf.valueChanged.connect(self._on_conf_change)
        cg.addWidget(self._slider_conf)

        # 프리셋 버튼
        preset_row = QHBoxLayout(); preset_row.setSpacing(4)
        for label, val in [("낮음\n0.30", 30), ("보통\n0.50", 50), ("높음\n0.70", 70)]:
            b = QPushButton(label)
            b.setFixedHeight(36)
            b.setStyleSheet(f"""
                QPushButton {{
                    background:{C_BTN_DARK}; color:{C_SUBTEXT};
                    border:1px solid {C_BORDER}; border-radius:5px;
                    font-family:Consolas; font-size:11px;
                }}
                QPushButton:hover {{ background:#2D333B; color:{C_TEXT}; }}
            """)
            b.clicked.connect(lambda _, v=val: self._slider_conf.setValue(v))
            preset_row.addWidget(b)
        cg.addLayout(preset_row)
        lay.addWidget(conf_grp)

        # ── 카메라 선택 ───────────────────────────────────────
        cam_grp = self._mk_group("카메라 선택", inner=True)
        camg = QVBoxLayout(cam_grp); camg.setSpacing(6)

        self._btn_cam_top = self._mk_btn("▲  상단 카메라  /rgb", C_ACCENT, "#000", h=36, small=True)
        self._btn_cam_bot = self._mk_btn("▼  하단 카메라  /rgb_pb", C_BTN_DARK, C_SUBTEXT, h=36, small=True)
        self._btn_cam_top.clicked.connect(lambda: self._on_cam_select("top"))
        self._btn_cam_bot.clicked.connect(lambda: self._on_cam_select("bottom"))
        camg.addWidget(self._btn_cam_top)
        camg.addWidget(self._btn_cam_bot)
        lay.addWidget(cam_grp)

        # ── 비전 실시간 지표 ──────────────────────────────────
        metric_grp = self._mk_group("실시간 지표", inner=True)
        mg = QVBoxLayout(metric_grp); mg.setSpacing(5)
        self._s_fps     = self._stat_row(mg, "감지 FPS",  "—")
        self._s_det_cnt = self._stat_row(mg, "누적 감지", "0")
        self._s_cam_fps = self._stat_row(mg, "카메라 FPS","—")
        lay.addWidget(metric_grp)

        lay.addStretch()

        # ── 수동 존 오버라이드 ────────────────────────────────
        ov_grp = self._mk_group("수동 존 오버라이드", inner=True)
        og = QVBoxLayout(ov_grp); og.setSpacing(5)
        ov_note = QLabel("자동 분류 오류 시 사용")
        ov_note.setStyleSheet(f"color:{C_SUBTEXT}; font-size:11px;")
        og.addWidget(ov_note)
        ov_row = QHBoxLayout(); ov_row.setSpacing(5)
        btn_ov_a = self._mk_btn("→ ZONE A", C_GREEN,  "#000", h=32, small=True)
        btn_ov_b = self._mk_btn("→ ZONE B", C_PURPLE, "#fff", h=32, small=True)
        btn_ov_a.clicked.connect(lambda: self._manual_zone("ZONE_A"))
        btn_ov_b.clicked.connect(lambda: self._manual_zone("ZONE_B"))
        ov_row.addWidget(btn_ov_a); ov_row.addWidget(btn_ov_b)
        og.addLayout(ov_row)
        lay.addWidget(ov_grp)
        return w

    # ── 진단 탭 ───────────────────────────────────────────────
    def _mk_diag_tab(self):
        from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem
        w = QWidget()
        lay = QVBoxLayout(w); lay.setSpacing(8); lay.setContentsMargins(10,10,10,10)

        # ── 전체 상태 배너 ─────────────────────────────────────
        self._diag_overall = QLabel("● 대기 중")
        self._diag_overall.setAlignment(Qt.AlignCenter)
        self._diag_overall.setFixedHeight(34)
        self._diag_overall.setStyleSheet(f"""
            QLabel {{
                background:{C_BTN_DARK}; color:{C_SUBTEXT};
                border-radius:6px; font-family:Consolas;
                font-size:14px; font-weight:bold; letter-spacing:1px;
            }}
        """)
        lay.addWidget(self._diag_overall)

        # ── 파이프라인 다이어그램 (SVG-like Canvas) ─────────────
        from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsTextItem
        from PyQt5.QtGui import QPen, QBrush, QPainterPath, QPainter
        from PyQt5.QtCore import QRectF, QPointF

        self._diag_scene = QGraphicsScene()
        self._diag_view  = QGraphicsView(self._diag_scene)
        self._diag_view.setFixedHeight(180)
        self._diag_view.setRenderHint(QPainter.Antialiasing)
        self._diag_view.setStyleSheet(f"""
            QGraphicsView {{
                background: #0D1117;
                border: 1px solid {C_BORDER};
                border-radius: 8px;
            }}
        """)
        self._diag_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._diag_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._draw_pipeline_diagram()
        lay.addWidget(self._diag_view)

        # ── 메트릭 카드 4개 ────────────────────────────────────
        metrics_row = QHBoxLayout(); metrics_row.setSpacing(6)
        self._m_cam_fps  = self._mk_metric_card(metrics_row, "카메라 FPS",  "—", C_ACCENT)
        self._m_det_fps  = self._mk_metric_card(metrics_row, "감지 FPS",   "—", C_GREEN)
        self._m_qr_cnt   = self._mk_metric_card(metrics_row, "QR 인식 수", "0", C_PURPLE)
        self._m_ok_cnt   = self._mk_metric_card(metrics_row, "정상 토픽",  "0/6", C_GREEN)
        lay.addLayout(metrics_row)

        # ── 토픽 상세 테이블 ───────────────────────────────────
        tbl_grp = self._mk_group("토픽 상세 상태", inner=True)
        tg = QVBoxLayout(tbl_grp); tg.setSpacing(3); tg.setContentsMargins(6,8,6,6)

        # 헤더
        hdr = QWidget(); hdr.setStyleSheet(f"background:{C_BTN_DARK}; border-radius:4px;")
        hh = QHBoxLayout(hdr); hh.setContentsMargins(8,3,8,3)
        for txt, w_ in [("토픽",150),("노드",90),("경과",50),("Hz",40),("상태",36)]:
            l = QLabel(txt); l.setFixedWidth(w_)
            l.setStyleSheet(f"color:{C_SUBTEXT}; font-size:11px; font-family:Consolas; font-weight:bold;")
            hh.addWidget(l)
        tg.addWidget(hdr)

        self._diag_rows = {}
        topics_info = [
            ("/rgb/compressed",              "image_transport", C_ACCENT),
            ("/parcel_detections/annotated", "parcel_detector", C_GREEN),
            ("/parcel_with_qr",              "parcel_detector", C_GREEN),
            ("/parcel_no_label",             "parcel_detector", C_GREEN),
            ("/qr_crop_image",               "qr_decoder_node", C_PURPLE),
            ("/qr_code",                     "qr_decoder_node", C_PURPLE),
        ]
        for topic, node, color in topics_info:
            row = self._mk_diag_row2(tg, topic, node, color)
            self._diag_rows[topic] = row

        lay.addWidget(tbl_grp)

        # 갱신 시각
        self._diag_last_update = QLabel("마지막 갱신: —")
        self._diag_last_update.setStyleSheet(f"color:{C_SUBTEXT}; font-size:11px; font-family:Consolas;")
        self._diag_last_update.setAlignment(Qt.AlignRight)
        lay.addWidget(self._diag_last_update)
        return w

    def _mk_metric_card(self, layout, label, val, color):
        card = QWidget()
        card.setStyleSheet(f"""
            QWidget {{
                background:{C_PANEL};
                border:1px solid {C_BORDER};
                border-radius:8px;
            }}
        """)
        v = QVBoxLayout(card); v.setContentsMargins(10,8,10,8); v.setSpacing(2)
        val_lbl = QLabel(val)
        val_lbl.setAlignment(Qt.AlignLeft)
        val_lbl.setStyleSheet(f"color:{color}; font-size:20px; font-weight:bold; font-family:Consolas;")
        name_lbl = QLabel(label)
        name_lbl.setStyleSheet(f"color:{C_SUBTEXT}; font-size:11px; font-family:Consolas;")
        v.addWidget(val_lbl); v.addWidget(name_lbl)
        layout.addWidget(card, stretch=1)
        return val_lbl

    def _draw_pipeline_diagram(self):
        from PyQt5.QtGui import QPen, QBrush, QColor, QPainterPath, QFont
        from PyQt5.QtCore import QRectF, Qt as QtCore_Qt
        sc = self._diag_scene
        sc.clear()

        W, H = 640, 170
        sc.setSceneRect(0, 0, W, H)

        C_NODE  = QColor("#161B22")
        C_BDR   = QColor("#30363D")
        C_TXT   = QColor("#E6EDF3")
        C_SUB   = QColor("#8B949E")
        C_CYAN  = QColor("#00D9FF")
        C_GRN   = QColor("#3FB950")
        C_PRP   = QColor("#A371F7")
        C_YLW   = QColor("#E3B341")
        C_BG    = QColor("#0D1117")
        C_DASH  = QColor("#30363D")

        def node(x,y,w,h,title,sub="",bdr_color=None):
            bc = QColor(bdr_color) if bdr_color else C_BDR
            r = sc.addRect(QRectF(x,y,w,h), QPen(bc,0.8), QBrush(C_NODE))
            r.setFlag(r.ItemIsSelectable, False)
            t = sc.addText(title)
            t.setDefaultTextColor(C_TXT)
            f = QFont("Consolas",7); f.setBold(True); t.setFont(f)
            t.setPos(x + w/2 - t.boundingRect().width()/2, y+6)
            if sub:
                s = sc.addText(sub)
                s.setDefaultTextColor(C_SUB)
                fs = QFont("Consolas",6); s.setFont(fs)
                s.setPos(x + w/2 - s.boundingRect().width()/2, y+20)

        def arrow(x1,y1,x2,y2,color):
            pen = QPen(QColor(color),1.2)
            pen.setStyle(QtCore_Qt.DashLine)
            sc.addLine(x1,y1,x2,y2,pen)
            # 화살표 머리
            path = QPainterPath()
            import math
            dx,dy = x2-x1, y2-y1
            ang = math.atan2(dy,dx)
            sz=6
            p1x = x2 - sz*math.cos(ang-0.4)
            p1y = y2 - sz*math.sin(ang-0.4)
            p2x = x2 - sz*math.cos(ang+0.4)
            p2y = y2 - sz*math.sin(ang+0.4)
            path.moveTo(p1x,p1y); path.lineTo(x2,y2); path.lineTo(p2x,p2y)
            sc.addPath(path, QPen(QColor(color),1.2), QBrush(QtCore_Qt.NoBrush))

        def dot_status(x,y,color):
            d = sc.addEllipse(QRectF(x-4,y-4,8,8), QPen(QtCore_Qt.NoPen), QBrush(QColor(color)))
            return d

        def small_txt(x,y,txt,color):
            t = sc.addText(txt)
            t.setDefaultTextColor(QColor(color))
            f = QFont("Consolas",6); t.setFont(f)
            t.setPos(x,y)
            return t

        # ── Isaac Sim PC ──────────────────────────────────────
        sc.addRect(QRectF(5,10,108,150), QPen(C_DASH,0.5,QtCore_Qt.DashLine), QBrush(QtCore_Qt.NoBrush))
        small_txt(20,2,"IsaacSim05","#8B949E")
        node(10,20,98,40,"Isaac Sim","시뮬레이션")
        node(10,68,98,40,"RealSense D455","/rgb (raw)")
        self._diag_dot_isaac = dot_status(104,88, C_SUBTEXT)

        # ── Vision PC ─────────────────────────────────────────
        sc.addRect(QRectF(125,5,510,160), QPen(C_DASH,0.5,QtCore_Qt.DashLine), QBrush(QtCore_Qt.NoBrush))
        small_txt(280,0,"Vision PC (taehwan)","#8B949E")

        # image_transport
        node(132,55,115,45,"image_transport","raw→/rgb/compressed","#00D9FF")
        self._diag_dot_imgtr = dot_status(243,78, C_GREEN)

        # parcel_detector
        node(268,18,118,48,"parcel_detector","YOLO11n","#3FB950")
        self._diag_dot_det = dot_status(382,42, C_GREEN)

        # qr_decoder
        node(268,90,118,44,"qr_decoder_node","pyzbar","#A371F7")
        self._diag_dot_qr = dot_status(382,112, C_GREEN)

        # GUI
        node(402,35,118,125,"Control GUI","parcel_gui_node","#00D9FF")
        small_txt(408,55,"/rgb/compressed","#00D9FF")
        small_txt(408,67,"/parcel_detections","#3FB950")
        small_txt(408,79,"/qr_code","#A371F7")
        small_txt(408,97,"→ /simulation_control","#E3B341")

        # ── 화살표 ────────────────────────────────────────────
        arrow(108,88, 130,78, "#00D9FF")       # Isaac→imgtr
        arrow(247,70, 266,35, "#00D9FF")       # imgtr→detector
        arrow(247,86, 266,108,"#00D9FF")       # imgtr→qr
        arrow(386,42, 400,65, "#3FB950")       # detector→GUI
        arrow(386,112,400,100,"#A371F7")       # qr→GUI
        # GUI→Isaac (제어)
        pen_ctrl = QPen(QColor("#E3B341"),0.8); pen_ctrl.setStyle(QtCore_Qt.DotLine)
        sc.addLine(402,120, 60,120, pen_ctrl)
        sc.addLine(60,120,  60,112, pen_ctrl)

        # 범례
        small_txt(132,148,"━ 카메라","#00D9FF")
        small_txt(212,148,"━ 감지결과","#3FB950")
        small_txt(302,148,"━ QR","#A371F7")
        small_txt(362,148,"··· 제어","#E3B341")

        # 노드 dot 참조 저장
        self._node_rows = {
            "image_transport": self._diag_dot_imgtr,
            "parcel_detector":  self._diag_dot_det,
            "qr_decoder_node":  self._diag_dot_qr,
            "isaac_sim":        self._diag_dot_isaac,
        }

    def _mk_diag_row2(self, layout, topic, node, topic_color):
        row_w = QWidget()
        rh = QHBoxLayout(row_w); rh.setContentsMargins(8,4,8,4); rh.setSpacing(0)

        dot = QLabel("●")
        dot.setFixedWidth(14)
        dot.setStyleSheet(f"color:{C_SUBTEXT}; font-size:14px;")

        name = QLabel(topic)
        name.setFixedWidth(150)
        name.setStyleSheet(f"color:{topic_color}; font-size:11px; font-family:Consolas; font-weight:bold;")

        nd = QLabel(node)
        nd.setFixedWidth(90)
        nd.setStyleSheet(f"color:{C_SUBTEXT}; font-size:11px; font-family:Consolas;")

        el = QLabel("—")
        el.setFixedWidth(50)
        el.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        el.setStyleSheet(f"color:{C_SUBTEXT}; font-size:11px; font-family:Consolas;")

        hz = QLabel("—")
        hz.setFixedWidth(40)
        hz.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hz.setStyleSheet(f"color:{C_SUBTEXT}; font-size:11px; font-family:Consolas;")

        st = QLabel("—")
        st.setFixedWidth(36)
        st.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        st.setStyleSheet(f"color:{C_SUBTEXT}; font-size:11px; font-family:Consolas; font-weight:bold;")

        rh.addWidget(dot); rh.addWidget(name); rh.addWidget(nd)
        rh.addWidget(el);  rh.addWidget(hz);   rh.addWidget(st)
        layout.addWidget(row_w)
        return {"dot": dot, "elapsed": el, "hz": hz, "status": st}

    def _refresh_diag(self):
        """1초마다 토픽 수신 상태 갱신"""
        from PyQt5.QtGui import QBrush, QColor
        from PyQt5.QtCore import Qt as QtCore_Qt
        now = time.time()
        all_ok  = True
        any_err = False
        ok_count = 0

        topic_to_nodes = {
            "/rgb/compressed":              "image_transport",
            "/parcel_detections/annotated": "parcel_detector",
            "/parcel_with_qr":              "parcel_detector",
            "/parcel_no_label":             "parcel_detector",
            "/qr_crop_image":               "qr_decoder_node",
            "/qr_code":                     "qr_decoder_node",
        }
        node_health = {"image_transport": True, "parcel_detector": True,
                       "qr_decoder_node": True, "isaac_sim": (self._state == "running")}

        for topic, row in self._diag_rows.items():
            last    = self._topic_last_recv.get(topic)
            timeout = self._topic_timeout.get(topic, 3.0)

            if self._state != "running":
                status = "idle"
            elif last is None:
                status = "error"
            else:
                elapsed_sec = now - last
                if elapsed_sec < timeout:        status = "ok"
                elif elapsed_sec < timeout * 2: status = "warn"
                else:                            status = "error"

            if status == "ok":
                color = C_GREEN;  elapsed_txt = f"{now-last:.1f}s"
                hz_txt = f"{int(20+10*(timeout/(now-last+0.01)))}Hz"
                ok_count += 1
            elif status == "warn":
                color = C_ORANGE; elapsed_txt = f"{now-last:.0f}s ⚠"; hz_txt = "—"
                all_ok = False
            elif status == "error":
                color = C_RED;    elapsed_txt = "없음 ✖";              hz_txt = "—"
                all_ok = False; any_err = True
                nd = topic_to_nodes.get(topic)
                if nd: node_health[nd] = False
            else:
                color = C_SUBTEXT; elapsed_txt = "—"; hz_txt = "—"

            st_txt = {"ok":"OK","warn":"WARN","error":"ERR","idle":"—"}.get(status,"—")
            row["dot"].setStyleSheet(f"color:{color}; font-size:14px;")
            row["elapsed"].setText(elapsed_txt)
            row["elapsed"].setStyleSheet(f"color:{color}; font-size:11px; font-family:Consolas;")
            row["hz"].setText(hz_txt)
            row["hz"].setStyleSheet(f"color:{color}; font-size:11px; font-family:Consolas;")
            row["status"].setText(st_txt)
            row["status"].setStyleSheet(f"color:{color}; font-size:11px; font-family:Consolas; font-weight:bold;")

        # 다이어그램 노드 점 색상 업데이트
        for node_id, dot_item in self._node_rows.items():
            if hasattr(dot_item, "setBrush"):
                ok = node_health.get(node_id, True)
                dot_item.setBrush(QBrush(QColor(C_GREEN if ok else C_RED)))

        # 메트릭 카드
        cam_fps = self._cam_conv.get_fps()
        self._m_cam_fps.setText(f"{cam_fps:.0f}" if cam_fps > 0 else "—")
        self._m_det_fps.setText(str(self._s_det_cnt.text()))
        self._m_qr_cnt.setText(str(self._qr_count))
        self._m_ok_cnt.setText(f"{ok_count}/6")
        ok_color = C_GREEN if ok_count == 6 else (C_ORANGE if ok_count >= 3 else C_RED)
        self._m_ok_cnt.setStyleSheet(f"color:{ok_color}; font-size:20px; font-weight:bold; font-family:Consolas;")

        # 전체 배너
        if self._state != "running":
            self._diag_overall.setText("●  시스템 미실행 — 시작 버튼을 누르세요")
            self._diag_overall.setStyleSheet(f"QLabel {{ background:{C_BTN_DARK}; color:{C_SUBTEXT}; border-radius:6px; font-family:Consolas; font-size:14px; font-weight:bold; letter-spacing:1px; }}")
        elif any_err:
            self._diag_overall.setText("✖  연결 오류 — 토픽 확인 필요")
            self._diag_overall.setStyleSheet(f"QLabel {{ background:{C_RED}; color:#fff; border-radius:6px; font-family:Consolas; font-size:14px; font-weight:bold; letter-spacing:1px; }}")
        elif not all_ok:
            self._diag_overall.setText("⚠  일부 토픽 지연 감지")
            self._diag_overall.setStyleSheet(f"QLabel {{ background:{C_ORANGE}; color:#000; border-radius:6px; font-family:Consolas; font-size:14px; font-weight:bold; letter-spacing:1px; }}")
        else:
            self._diag_overall.setText("✔  모든 시스템 정상")
            self._diag_overall.setStyleSheet(f"QLabel {{ background:{C_GREEN}; color:#000; border-radius:6px; font-family:Consolas; font-size:14px; font-weight:bold; letter-spacing:1px; }}")

        self._diag_last_update.setText(
            f"마지막 갱신: {datetime.datetime.now().strftime('%H:%M:%S')}")
    def _mk_video_panel(self):
        box = self._mk_group("실시간 카메라 피드")
        from PyQt5.QtWidgets import QGridLayout
        grid = QGridLayout(box)
        grid.setSpacing(8); grid.setContentsMargins(8,12,8,8)

        self._cam_conv       = CameraView("컨베이어")
        self._cam_parcel     = CameraView("택배 감지")
        self._cam_annotated  = CameraView("YOLO 어노테이션")
        self._cam_qr_crop    = CameraView("QR 크롭")

        grid.addWidget(self._wrap_cam("컨베이어  /rgb/compressed",               self._cam_conv),      0, 0)
        grid.addWidget(self._wrap_cam("택배 감지  /parcel_with_qr",              self._cam_parcel),    0, 1)
        grid.addWidget(self._wrap_cam("YOLO 결과  /parcel_detections/annotated", self._cam_annotated), 1, 0)

        # QR 크롭 + 디코딩 결과를 나란히 배치
        qr_row = QWidget()
        qr_h = QHBoxLayout(qr_row); qr_h.setSpacing(8); qr_h.setContentsMargins(0,0,0,0)
        qr_h.addWidget(self._wrap_cam("QR 크롭  /qr_crop_image", self._cam_qr_crop), stretch=3)
        qr_h.addWidget(self._mk_qr_result_panel(), stretch=2)
        grid.addWidget(qr_row, 1, 1)

        return box

    def _mk_qr_result_panel(self) -> QWidget:
        """QR 디코딩 결과를 크게 표시하는 패널"""
        w = QWidget()
        v = QVBoxLayout(w); v.setSpacing(4); v.setContentsMargins(0,0,0,0)

        title = QLabel("QR 디코딩 결과")
        title.setStyleSheet(f"color:{C_SUBTEXT}; font-size:12px; font-family:Consolas;")
        v.addWidget(title)

        box = QWidget()
        box.setStyleSheet(f"""
            QWidget {{
                background:#10171E;
                border:1px solid {C_BORDER};
                border-radius:6px;
            }}
        """)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        inner = QVBoxLayout(box); inner.setSpacing(6)

        # 존 표시 (크게)
        self._lbl_zone_big = QLabel("—")
        self._lbl_zone_big.setAlignment(Qt.AlignCenter)
        self._lbl_zone_big.setStyleSheet(f"""
            color:{C_ACCENT};
            font-family:Consolas;
            font-size:42px;
            font-weight:bold;
        """)
        inner.addWidget(self._lbl_zone_big)

        # 구분선
        line = QFrame(); line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color:{C_BORDER};")
        inner.addWidget(line)

        # 마지막 인식 시각
        self._lbl_zone_time = QLabel("—")
        self._lbl_zone_time.setAlignment(Qt.AlignCenter)
        self._lbl_zone_time.setStyleSheet(f"color:{C_SUBTEXT}; font-family:Consolas; font-size:13px;")
        inner.addWidget(self._lbl_zone_time)

        # 누적 카운트
        cnt_row = QHBoxLayout()
        self._lbl_za_big = self._zone_cnt_lbl("A", C_GREEN)
        self._lbl_zb_big = self._zone_cnt_lbl("B", C_PURPLE)
        cnt_row.addWidget(self._lbl_za_big)
        cnt_row.addWidget(self._lbl_zb_big)
        inner.addLayout(cnt_row)

        v.addWidget(box)
        return w

    def _zone_cnt_lbl(self, zone, color) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w); v.setSpacing(2); v.setContentsMargins(4,4,4,4)
        lbl = QLabel(f"ZONE {zone}")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(f"color:{C_SUBTEXT}; font-size:12px; font-family:Consolas;")
        cnt = QLabel("0")
        cnt.setAlignment(Qt.AlignCenter)
        cnt.setStyleSheet(f"color:{color}; font-size:22px; font-weight:bold; font-family:Consolas;")
        v.addWidget(lbl); v.addWidget(cnt)
        setattr(self, f"_lbl_z{'a' if zone=='A' else 'b'}_big_cnt", cnt)
        return w

    def _wrap_cam(self, title, cam):
        w = QWidget(); v = QVBoxLayout(w); v.setSpacing(4); v.setContentsMargins(0,0,0,0)
        lbl = QLabel(title)
        lbl.setStyleSheet(f"color:{C_SUBTEXT}; font-size:12px; font-family:Consolas;")
        v.addWidget(lbl); v.addWidget(cam)
        return w

    # ── 로그 패널 ─────────────────────────────────────────────
    def _mk_log_panel(self):
        box = self._mk_group("이벤트 로그")
        lay = QVBoxLayout(box); lay.setContentsMargins(8,8,8,8)
        self._log = LogWidget()
        lay.addWidget(self._log)
        self._log.log("시스템 초기화 완료. 자동 시작 중...", "INFO")
        return box

    def _emit_log(self, msg: str, level: str = "INFO"):
        """메인 로그 패널 + LogWindow(열려있으면)에 동시 전달"""
        self._log.log(msg, level)
        if hasattr(self, '_log_window') and self._log_window.isVisible():
            self._log_window.append(msg, level)

    # ── 캡처 패널 ─────────────────────────────────────────────
    def _mk_capture_panel(self):
        box = self._mk_group("캡처 이미지")
        lay = QVBoxLayout(box); lay.setContentsMargins(8,8,8,8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{ border:none; background:transparent; }}
            QScrollBar:horizontal {{ background:{C_PANEL}; height:5px; border-radius:2px; }}
            QScrollBar::handle:horizontal {{ background:{C_BORDER}; border-radius:2px; }}
        """)
        inner = QWidget()
        self._thumb_lay = QHBoxLayout(inner)
        self._thumb_lay.setSpacing(6); self._thumb_lay.setContentsMargins(0,0,0,0)
        self._thumb_lay.addStretch()
        scroll.setWidget(inner)
        lay.addWidget(scroll)
        return box

    def _mk_statusbar(self):
        bar = QStatusBar()
        bar.setStyleSheet(f"""
            QStatusBar {{
                background:{C_PANEL}; border-top:1px solid {C_BORDER};
                color:{C_SUBTEXT}; font-size:13px; font-family:Consolas; padding:2px 8px;
            }}
        """)
        self.setStatusBar(bar)
        self._sb = bar
        bar.showMessage(f"저장: {SAVE_DIR}  |  ROS_DOMAIN_ID={DOMAIN_ID}  |  Conf=0.50")

    # ──────────────────────────────────────────────────────────
    #  버튼 이벤트 — 시스템
    # ──────────────────────────────────────────────────────────
    def _on_start(self):
        if self._state == "running": return
        self._state = "running"
        self._badge.set_state("running")
        self._btn_start.setEnabled(False)
        self._lbl_sim_state.setText("● 실행 중")
        self._lbl_sim_state.setStyleSheet(
            f"color:{C_GREEN}; font-family:Consolas; font-size:14px; font-weight:bold; padding:6px;")
        # ROS2 구독/퍼블리시 스레드 시작
        if ROS2_AVAILABLE and not self._ros.isRunning():
            self._ros.start()
        # Isaac Sim에 play 신호 전송 — 스레드 초기화 대기 후 800ms 딜레이
        QTimer.singleShot(800, self._send_play)
        self._sb.showMessage("▶ 재생 신호 전송 → /simulation_control: play")
        self._emit_log("▶ 재생 → /simulation_control: play", "CMD")

    def _send_play(self):
        self._ros.publish_sim_ctrl("play")

    def _launch_isaac(self):
        exe = os.environ.get("ISAAC_SIM_EXEC",
                             "/isaac-sim/runheadless.native.sh")
        if not Path(exe).exists():
            self._sb.showMessage(f"⚠ Isaac Sim 실행 파일 없음: {exe}")
            self._emit_log(f"Isaac Sim 경로 없음: {exe}", "ERR")
            return
        self._isaac_proc = subprocess.Popen([exe])
        self._sb.showMessage(f"Isaac Sim 실행됨 (PID {self._isaac_proc.pid})")
        self._emit_log(f"Isaac Sim PID {self._isaac_proc.pid}", "OK")

    def _on_estop(self):
        self._state = "stopped"
        self._badge.set_state("stopped")
        self._btn_start.setEnabled(True)   # 재시작 가능하도록 유지
        self._lbl_sim_state.setText("● 긴급 정지")
        self._lbl_sim_state.setStyleSheet(
            f"color:{C_RED}; font-family:Consolas; font-size:14px; font-weight:bold; padding:6px;")
        self._ros.publish_e_stop(True)
        self._ros.publish_sim_ctrl("pause")
        self._sb.showMessage("⬛ 긴급 정지 → /simulation_control: pause")
        self._emit_log("⬛ 긴급 정지 → /simulation_control: pause  |  /emergency_stop", "ERR")

    def _on_reset(self):
        if QMessageBox.question(self, "초기화", "시스템을 초기화하시겠습니까?",
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self._state = "idle"; self._badge.set_state("idle")
        self._btn_start.setEnabled(True)
        self._lbl_sim_state.setText("● 대기 중")
        self._lbl_sim_state.setStyleSheet(
            f"color:{C_SUBTEXT}; font-family:Consolas; font-size:14px; font-weight:bold; padding:6px;")
        self._lbl_za_cnt.setText("0")
        self._lbl_zb_cnt.setText("0")
        self._parcel_count = self._qr_count = self._zone_a_cnt = self._zone_b_cnt = self._no_label_cnt = 0
        for lbl, val in [(self._s_total,"0"),(self._s_qr,"0"),(self._s_za,"0"),(self._s_zb,"0"),(self._s_nl,"0")]:
            lbl.setText(val)
        self._lbl_zone.setText("—")
        self._lbl_zone_big.setText("—")
        self._lbl_zone_time.setText("—")
        self._lbl_zone_big.setStyleSheet(f"color:{C_ACCENT}; font-family:Consolas; font-size:42px; font-weight:bold;")
        self._lbl_za_big_cnt.setText("0")
        self._lbl_zb_big_cnt.setText("0")
        while self._thumb_lay.count() > 1:
            it = self._thumb_lay.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        self._ros.publish_e_stop(False)
        self._ros.publish_reset()
        self._ros.publish_sim_ctrl("reset")
        # QR 베스트샷 버퍼 초기화
        self._qr_bestshot_timer.stop()
        self._qr_bestshot_buf    = []
        self._qr_bestshot_active = False
        self._pkg_bestshot_timer.stop()
        self._pkg_bestshot_buf   = []
        self._pkg_bestshot_active = False
        self._cached_conv_frame  = None
        self._sb.showMessage("↺ 초기화 완료")
        self._emit_log("시스템 초기화 → /simulation_control: reset", "OK")

    # ──────────────────────────────────────────────────────────
    #  버튼 이벤트 — 비전
    # ──────────────────────────────────────────────────────────
    def _on_det_toggle(self, state):
        self._det_enabled = bool(state)
        self._ros.publish_det_enable(self._det_enabled)
        txt = "감지 ON" if self._det_enabled else "감지 OFF"
        col = C_GREEN if self._det_enabled else C_RED
        self._lbl_det_indicator.setText(f"● {txt}")
        self._lbl_det_indicator.setStyleSheet(f"color:{col}; font-family:Consolas; font-size:13px;")
        level = "VISION" if self._det_enabled else "WARN"
        self._emit_log(f"YOLO 감지 {'활성화' if self._det_enabled else '비활성화'} → {DETECT_EN_TOPIC}", level)

    def _on_save_toggle(self, state):
        self._save_enabled = bool(state)
        self._ros.publish_save_enable(self._save_enabled)
        self._emit_log(f"이미지 저장 {'ON' if self._save_enabled else 'OFF'} → {SAVE_EN_TOPIC}", "VISION")

    def _on_conf_change(self, val):
        self._conf_val = val / 100.0
        self._lbl_conf_val.setText(f"{self._conf_val:.2f}")
        self._ros.publish_conf(self._conf_val)
        self._sb.showMessage(
            f"저장: {SAVE_DIR}  |  ROS_DOMAIN_ID={DOMAIN_ID}  |  Conf={self._conf_val:.2f}")
        self._emit_log(f"Conf threshold → {self._conf_val:.2f}  ({CONF_TOPIC})", "VISION")

    def _on_cam_select(self, cam: str):
        self._cam_top = (cam == "top")
        self._ros.publish_cam_select(cam)
        self._lbl_cam_indicator.setText("CAM: 상단" if self._cam_top else "CAM: 하단")
        # 버튼 하이라이트
        self._btn_cam_top.setStyleSheet(
            self._btn_style(C_ACCENT, "#000", small=True) if self._cam_top
            else self._btn_style(C_BTN_DARK, C_SUBTEXT, small=True))
        self._btn_cam_bot.setStyleSheet(
            self._btn_style(C_ACCENT, "#000", small=True) if not self._cam_top
            else self._btn_style(C_BTN_DARK, C_SUBTEXT, small=True))
        self._emit_log(f"카메라 선택: {'상단 /rgb' if self._cam_top else '하단 /rgb_pb'} → {CAM_SEL_TOPIC}", "VISION")

    def _manual_zone(self, zone: str):
        self._ros.publish_cam_select(zone)   # 별도 오버라이드 토픽으로 교체 가능
        self._lbl_zone.setText(zone)
        self._emit_log(f"수동 존 오버라이드: {zone}", "WARN")

    # ──────────────────────────────────────────────────────────
    #  ROS2 콜백
    # ──────────────────────────────────────────────────────────
    def _record_topic(self, topic: str, nbytes: int = 0):
        """토픽 수신 시각 + Hz 윈도우 + BW 누적"""
        now = time.time()
        last = self._topic_last_recv.get(topic)
        if last is not None:
            interval = now - last
            if 0 < interval < 10:
                win = self._topic_hz_window[topic]
                win.append(interval)
                if len(win) > 30: win.pop(0)
        self._topic_last_recv[topic] = now
        self._topic_recv_cnt[topic]  = self._topic_recv_cnt.get(topic, 0) + 1
        self._topic_recv_bw[topic]   = self._topic_recv_bw.get(topic, 0)  + nbytes

    def _on_conveyor(self, data):
        self._record_topic("/rgb/compressed", len(data))
        self._cam_conv.update_from_compressed(data)
        # 최신 프레임 캐시 (bbox 크롭에 사용)
        if CV2_AVAILABLE:
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                self._cached_conv_frame = img

    def _on_parcel_qr(self, json_str: str):
        """Detection2DArray JSON 수신 → package bbox로 컨베이어 프레임 크롭 → 카메라 뷰 표시"""
        self._record_topic("/parcel_with_qr", len(json_str))
        if not CV2_AVAILABLE: return
        frame = getattr(self, '_cached_conv_frame', None)
        if frame is None: return

        try:
            dets = json.loads(json_str)
        except Exception:
            return

        # package 클래스 bbox 중 가장 큰 것 선택
        pkg_dets = [d for d in dets if d.get('cls') == 'package']
        if not pkg_dets:
            return
        best = max(pkg_dets, key=lambda d: d['w'] * d['h'])

        h, w = frame.shape[:2]
        cx, cy  = best['cx'], best['cy']
        bw, bh  = best['w'],  best['h']
        pad     = 20
        x1 = max(0,  int(cx - bw/2 - pad))
        y1 = max(0,  int(cy - bh/2 - pad))
        x2 = min(w,  int(cx + bw/2 + pad))
        y2 = min(h,  int(cy + bh/2 + pad))

        if x2 <= x1 or y2 <= y1:
            return

        cropped = frame[y1:y2, x1:x2].copy()

        # bbox 테두리 표시
        cv2.rectangle(cropped, (pad, pad),
                      (cropped.shape[1]-pad, cropped.shape[0]-pad),
                      (0, 200, 80), 2)
        conf = best.get('conf', 0)
        cv2.putText(cropped, f"package {conf:.2f}",
                    (pad+4, pad+20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 200, 80), 1)

        self._cam_parcel.update_from_cv(cropped)

        if self._save_enabled:
            # bestshot 방식: 1.5초 윈도우 내 가장 큰 bbox 1장만 저장
            if not CV2_AVAILABLE:
                pass
            else:
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                area = best['w'] * best['h']
                score = sharpness * area
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                if not self._pkg_bestshot_active:
                    self._pkg_bestshot_active = True
                    self._pkg_bestshot_buf    = []
                    self._pkg_bestshot_buf.append((score, cropped.copy(), ts))
                    self._pkg_bestshot_timer.start(int(self._BESTSHOT_WINDOW * 1000))
                else:
                    self._pkg_bestshot_buf.append((score, cropped.copy(), ts))

    def _on_parcel_noq(self, data):
        self._record_topic("/parcel_no_label", len(data))
        if not CV2_AVAILABLE: return
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: return
        # NO_LABEL 카운트 증가
        self._no_label_cnt += 1
        self._s_nl.setText(str(self._no_label_cnt))
        self._emit_log(f"라벨 없는 택배 감지 (누적 {self._no_label_cnt}건)", "WARN")
        if self._save_enabled:
            self._save_img(img, suffix="no_label")

    def _on_annotated(self, h, w, encoding, data):
        self._record_topic("/parcel_detections/annotated", len(data))
        img = self._raw_msg_to_cv(h, w, encoding, data)
        if img is not None:
            self._cam_annotated.update_from_cv(img)

    def _on_qr_crop(self, h, w, encoding, data):
        self._record_topic("/qr_crop_image", len(data))
        img = self._raw_msg_to_cv(h, w, encoding, data)
        if img is None:
            return
        self._cam_qr_crop.update_from_cv(img)
        if not self._save_enabled or not CV2_AVAILABLE:
            return
        gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if not self._qr_bestshot_active:
            self._qr_bestshot_active = True
            self._qr_bestshot_buf    = []
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._qr_bestshot_buf.append((sharpness, img.copy(), ts))
            self._qr_bestshot_timer.start(int(self._BESTSHOT_WINDOW * 1000))
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._qr_bestshot_buf.append((sharpness, img.copy(), ts))

    def _flush_pkg_bestshot(self):
        """택배 수집 윈도우 종료 → 가장 선명하고 큰 프레임 1장 저장"""
        if not self._pkg_bestshot_buf:
            self._pkg_bestshot_active = False
            return
        best_score, best_img, best_ts = max(
            self._pkg_bestshot_buf, key=lambda x: x[0])
        self._save_img(best_img)
        self._emit_log(
            f"택배 베스트샷 저장 (후보 {len(self._pkg_bestshot_buf)}장 중 최고)", "OK")
        self._pkg_bestshot_buf    = []
        self._pkg_bestshot_active = False

    def _flush_qr_bestshot(self):
        """수집 윈도우 종료 → 가장 선명한 프레임 1장 저장"""
        if not self._qr_bestshot_buf:
            self._qr_bestshot_active = False
            return

        # 선명도 기준 정렬 후 최상위 1장 선택
        best_sharpness, best_img, best_ts = max(
            self._qr_bestshot_buf, key=lambda x: x[0])

        fn = f"qr_{best_ts}.jpg"
        cv2.imwrite(str(QR_SAVE_DIR / fn), best_img)

        rgb  = cv2.cvtColor(best_img, cv2.COLOR_BGR2RGB)
        rh, rw = rgb.shape[:2]
        qimg = QImage(rgb.data, rw, rh, rw * 3, QImage.Format_RGB888)
        self._add_thumb(QPixmap.fromImage(qimg), fn, badge="QR")

        self._emit_log(
            f"QR 베스트샷 저장: {fn}  "
            f"(후보 {len(self._qr_bestshot_buf)}장 중 선명도 {best_sharpness:.0f})", "OK")

        # 버퍼 초기화
        self._qr_bestshot_buf    = []
        self._qr_bestshot_active = False

    @staticmethod
    def _raw_msg_to_cv(h, w, encoding, data):
        """sensor_msgs/Image raw 데이터를 OpenCV BGR 이미지로 변환"""
        if not CV2_AVAILABLE:
            return None
        enc = encoding.lower()
        try:
            if enc in ("rgb8", "rgb"):
                arr = np.frombuffer(data, np.uint8).reshape(h, w, 3)
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif enc in ("bgr8", "bgr"):
                return np.frombuffer(data, np.uint8).reshape(h, w, 3).copy()
            elif enc in ("rgba8",):
                arr = np.frombuffer(data, np.uint8).reshape(h, w, 4)
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif enc in ("bgra8",):
                arr = np.frombuffer(data, np.uint8).reshape(h, w, 4)
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            elif enc in ("mono8",):
                arr = np.frombuffer(data, np.uint8).reshape(h, w)
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif enc in ("16uc1", "mono16"):
                arr = np.frombuffer(data, np.uint16).reshape(h, w)
                arr8 = (arr / 256).astype(np.uint8)
                return cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)
            else:
                # 알 수 없는 인코딩 — rgb8 으로 시도
                arr = np.frombuffer(data, np.uint8).reshape(h, w, 3)
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _on_hub_state(self, data: str):
        """허브로부터 노드 종합 상태 수신 → 캐시 저장 + DiagWindow 갱신 트리거"""
        try:
            self._hub_state = json.loads(data)
        except Exception:
            return

        # DiagWindow가 열려있으면 즉시 갱신
        if hasattr(self, '_diag_window') and self._diag_window.isVisible():
            self._diag_window.refresh_from_hub(self._hub_state)

    def _on_hub_alert(self, data: str):
        """허브 경고 수신 → 로그 출력 + 진단 창 갱신"""
        try:
            alert = json.loads(data)
            level = alert.get('level', 'WARN')
            node  = alert.get('node', '?')
            msg   = alert.get('msg', '')
            self._emit_log(f'[HUB] {node}: {msg}', level)
            # ERR 수준이면 상태바에도 표시
            if level == 'ERR':
                self._sb.showMessage(f'⚠ 허브 경고 — {node}: {msg}')
        except Exception:
            pass

    def _on_qr_zone(self, zone):
        self._record_topic("/qr_code", len(zone.encode()))
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        # 왼쪽 패널 소형 표시
        self._lbl_zone.setText(zone)
        # 카메라 옆 대형 표시
        self._lbl_zone_big.setText(zone)
        self._lbl_zone_time.setText(f"인식 시각  {ts}")
        # 존 색상 구분
        color = C_GREEN if "A" in zone else C_PURPLE
        self._lbl_zone_big.setStyleSheet(f"""
            color:{color}; font-family:Consolas;
            font-size:42px; font-weight:bold;
        """)
        self._qr_count += 1; self._s_qr.setText(str(self._qr_count))
        if "A" in zone:
            self._zone_a_cnt += 1
            self._s_za.setText(str(self._zone_a_cnt))
            self._lbl_za_big_cnt.setText(str(self._zone_a_cnt))
            self._lbl_za_cnt.setText(str(self._zone_a_cnt))
        elif "B" in zone:
            self._zone_b_cnt += 1
            self._s_zb.setText(str(self._zone_b_cnt))
            self._lbl_zb_big_cnt.setText(str(self._zone_b_cnt))
            self._lbl_zb_cnt.setText(str(self._zone_b_cnt))
        self._emit_log(f"QR 인식: {zone}", "OK")

    def _on_fps(self, fps):
        self._s_fps.setText(f"{fps:.1f}")

    def _on_det_count(self, cnt):
        self._s_det_cnt.setText(str(cnt))

    def _refresh_fps_display(self):
        fps = self._cam_conv.get_fps()
        if fps > 0:
            self._s_cam_fps.setText(f"{fps:.1f}")

    # ──────────────────────────────────────────────────────────
    #  이미지 저장
    # ──────────────────────────────────────────────────────────
    def _save_img(self, img, suffix=""):
        self._parcel_count += 1; self._s_total.setText(str(self._parcel_count))
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tag = f"_{suffix}" if suffix else ""
        fn  = f"parcel_{ts}{tag}.jpg"
        cv2.imwrite(str(PKG_SAVE_DIR / fn), img)
        # QR 크롭은 /qr_crop_image 토픽에서 직접 수신하므로 수동 크롭 불필요
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
        self._add_thumb(QPixmap.fromImage(qimg), fn, badge="PKG")

    def _add_thumb(self, pm, name, badge="PKG"):
        th = ThumbnailWidget(name)
        th.set_image(pm, name)
        self._thumb_lay.insertWidget(self._thumb_lay.count()-1, th)

    # ──────────────────────────────────────────────────────────
    #  데모 모드
    # ──────────────────────────────────────────────────────────
    def _demo_frame(self):
        if not CV2_AVAILABLE: return
        h, w = 240, 320
        fr = np.random.randint(15, 40, (h, w, 3), np.uint8)
        x  = int((time.time() * 70) % (w+70)) - 70
        cv2.rectangle(fr, (x,75), (x+65,155), (60,100,160), -1)
        cv2.putText(fr,"PKG",(x+8,125),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,210,240),1)
        for i in range(0,w,40): cv2.line(fr,(i,0),(i,h),(30,50,30),1)
        if self._det_enabled:
            cv2.rectangle(fr,(x+5,85),(x+60,145),(0,200,80),1)
            cv2.putText(fr,f"{self._conf_val:.2f}",(x+5,82),
                        cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,200,80),1)
        self._cam_conv.update_from_cv(fr)

    # ──────────────────────────────────────────────────────────
    #  헬퍼
    # ──────────────────────────────────────────────────────────
    def _update_clock(self):
        self._clock_lbl.setText(datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    def _stat_row(self, lay, lbl, val):
        row = QHBoxLayout()
        l = QLabel(lbl); l.setStyleSheet(f"color:{C_SUBTEXT}; font-size:13px;")
        v = QLabel(val); v.setAlignment(Qt.AlignRight)
        v.setStyleSheet(f"color:{C_TEXT}; font-family:Consolas; font-size:15px; font-weight:bold;")
        row.addWidget(l); row.addWidget(v)
        lay.addLayout(row)
        return v

    def _mk_btn(self, text, bg, fg, h=44, small=False):
        b = QPushButton(text)
        b.setFixedHeight(h)
        b.setStyleSheet(self._btn_style(bg, fg, small))
        return b

    def _btn_style(self, bg, fg, small=False):
        fs = "11px" if small else "13px"
        return f"""
            QPushButton {{
                background:{bg}; color:{fg}; border:none; border-radius:6px;
                font-size:{fs}; font-weight:bold; font-family:Consolas; letter-spacing:1px;
            }}
            QPushButton:hover {{ border:1px solid rgba(255,255,255,0.15); }}
            QPushButton:pressed {{ opacity:0.7; }}
            QPushButton:disabled {{ background:#2D333B; color:#484F58; }}
        """

    def _mk_group(self, title="", inner=False):
        g = QGroupBox(title)
        bg = "#1C2128" if inner else C_PANEL
        g.setStyleSheet(f"""
            QGroupBox {{
                background:{bg}; border:1px solid {C_BORDER}; border-radius:8px;
                color:{C_SUBTEXT}; font-size:13px; font-family:Consolas;
                margin-top:8px; padding-top:6px;
            }}
            QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 4px; color:{C_SUBTEXT}; }}
        """)
        return g

    def _divider(self):
        f = QFrame(); f.setFrameShape(QFrame.HLine)
        f.setStyleSheet(f"color:{C_BORDER};"); return f

    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{C_BG}; color:{C_TEXT}; }}
            QMessageBox {{ background:{C_PANEL}; color:{C_TEXT}; }}
            QMessageBox QLabel {{ color:{C_TEXT}; }}
            QMessageBox QPushButton {{
                background:{C_BTN_DARK}; color:{C_TEXT};
                border:1px solid {C_BORDER}; border-radius:4px; padding:4px 12px;
            }}
            QScrollBar:vertical {{ background:{C_PANEL}; width:5px; border-radius:2px; }}
            QScrollBar::handle:vertical {{ background:{C_BORDER}; border-radius:2px; }}
        """)

    def _open_diag_window(self):
        if hasattr(self, '_diag_window') and self._diag_window.isVisible():
            self._diag_window.raise_(); self._diag_window.activateWindow(); return
        self._diag_window = DiagWindow(self)
        self._diag_window.show()

    def _open_log_window(self):
        if hasattr(self, '_log_window') and self._log_window.isVisible():
            self._log_window.raise_(); self._log_window.activateWindow(); return
        self._log_window = LogWindow(self)
        self._log_window.show()

    def _open_chart_window(self):
        if hasattr(self, '_chart_window') and self._chart_window.isVisible():
            self._chart_window.raise_(); self._chart_window.activateWindow(); return
        self._chart_window = ChartWindow(self)
        self._chart_window.show()

    def showEvent(self, e):
        super().showEvent(e)
        if not self._autostart_done:
            self._autostart_done = True
            QTimer.singleShot(300, self._on_start)

    def closeEvent(self, e):
        self._ros.stop()
        if self._isaac_proc and self._isaac_proc.poll() is None:
            self._isaac_proc.terminate()
        e.accept()


# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
#  진단 창 (독립 윈도우)
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
#  이벤트 로그 창 (독립 윈도우)
# ═══════════════════════════════════════════════════════════════
class ChartWindow(QMainWindow):
    """통계 그래프 창 — 도넛 차트 + 시계열 바 차트"""
    def __init__(self, parent):
        super().__init__(parent)
        self._p = parent
        self.setWindowTitle("통계 그래프 — Parcel Sorting Analytics")
        self.setMinimumSize(900, 620)
        self.setStyleSheet(f"QMainWindow,QWidget{{background:{C_BG};color:{C_TEXT};}}")

        cw = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw); root.setSpacing(10); root.setContentsMargins(14,10,14,10)

        # ── 상단 타이틀 + 새로고침 ────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("📊  분류 통계 현황")
        title.setStyleSheet(f"color:{C_ACCENT};font-size:15px;font-weight:bold;font-family:Consolas;letter-spacing:2px;")
        hdr.addWidget(title)
        hdr.addStretch()
        self._ts_lbl = QLabel("—")
        self._ts_lbl.setStyleSheet(f"color:{C_SUBTEXT};font-size:11px;font-family:Consolas;")
        hdr.addWidget(self._ts_lbl)
        root.addLayout(hdr)

        # ── 숫자 카드 5개 ─────────────────────────────────────
        cards = QHBoxLayout(); cards.setSpacing(8)
        self._cards = {}
        for key, lbl, col in [
            ("total",   "총 택배",    C_TEXT),
            ("zone_a",  "ZONE A",    C_GREEN),
            ("zone_b",  "ZONE B",    C_PURPLE),
            ("no_label","라벨 없음", C_ORANGE),
            ("qr",      "QR 인식",   C_ACCENT),
        ]:
            card = QWidget()
            card.setStyleSheet(f"""
                QWidget{{background:{C_PANEL};border:1px solid {C_BORDER};
                border-radius:10px;}}
            """)
            v = QVBoxLayout(card); v.setContentsMargins(14,12,14,12); v.setSpacing(4)
            val = QLabel("0")
            val.setAlignment(Qt.AlignCenter)
            val.setStyleSheet(f"color:{col};font-size:32px;font-weight:bold;font-family:Consolas;")
            name = QLabel(lbl)
            name.setAlignment(Qt.AlignCenter)
            name.setStyleSheet(f"color:{C_SUBTEXT};font-size:11px;font-family:Consolas;")
            v.addWidget(val); v.addWidget(name)
            cards.addWidget(card, stretch=1)
            self._cards[key] = val
        root.addLayout(cards)

        # ── 차트 영역 (좌: 도넛, 우: 시계열) ─────────────────
        chart_row = QHBoxLayout(); chart_row.setSpacing(10)

        # 도넛 차트
        donut_grp = QGroupBox("존별 분류 비율")
        donut_grp.setStyleSheet(f"""
            QGroupBox{{background:{C_PANEL};border:1px solid {C_BORDER};border-radius:8px;
              color:{C_SUBTEXT};font-size:11px;font-family:Consolas;margin-top:8px;padding-top:6px;}}
            QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 4px;}}
        """)
        dl = QVBoxLayout(donut_grp); dl.setContentsMargins(8,12,8,8)
        self._donut = DonutChart()
        self._donut.setMinimumSize(280, 280)
        dl.addWidget(self._donut)
        chart_row.addWidget(donut_grp, stretch=1)

        # 시계열 바 차트
        bar_grp = QGroupBox("시간대별 처리량 (최근 20회)")
        bar_grp.setStyleSheet(donut_grp.styleSheet())
        bl = QVBoxLayout(bar_grp); bl.setContentsMargins(8,12,8,8)
        self._bar = BarChart()
        self._bar.setMinimumSize(400, 280)
        bl.addWidget(self._bar)
        chart_row.addWidget(bar_grp, stretch=2)
        root.addLayout(chart_row)

        # ── 갱신 타이머 ────────────────────────────────────────
        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(1000)
        self._refresh()

    def _refresh(self):
        p = self._p
        total  = p._parcel_count
        za     = p._zone_a_cnt
        zb     = p._zone_b_cnt
        nl     = p._no_label_cnt
        qr     = p._qr_count

        # 카드 업데이트
        self._cards["total"].setText(str(total))
        self._cards["zone_a"].setText(str(za))
        self._cards["zone_b"].setText(str(zb))
        self._cards["no_label"].setText(str(nl))
        self._cards["qr"].setText(str(qr))

        # 도넛 차트
        self._donut.set_data([
            ("ZONE A",    za, C_GREEN),
            ("ZONE B",    zb, C_PURPLE),
            ("라벨 없음", nl, C_ORANGE),
        ])

        # 바 차트 — 현재 스냅샷 추가
        self._bar.add_snapshot(za, zb, nl)

        self._ts_lbl.setText(
            f"갱신: {datetime.datetime.now().strftime('%H:%M:%S')}"
            f"  |  총 처리: {total}건")

    def closeEvent(self, e):
        self._timer.stop(); e.accept()


class DonutChart(QWidget):
    """도넛 파이 차트 위젯"""
    def __init__(self):
        super().__init__()
        self._data = []   # (label, value, color)
        self.setMinimumSize(200, 200)

    def set_data(self, data):
        self._data = data
        self.update()

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont
        from PyQt5.QtCore import QRectF, Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h   = self.width(), self.height()
        margin = 30
        size   = min(w, h) - margin * 2
        x      = (w - size) / 2
        y      = (h - size) / 2
        rect   = QRectF(x, y, size, size)
        hole   = QRectF(x + size*0.28, y + size*0.28, size*0.44, size*0.44)

        total  = sum(v for _, v, _ in self._data) if self._data else 0

        # 배경
        painter.fillRect(self.rect(), QColor(C_PANEL))

        if total == 0:
            painter.setPen(QColor(C_SUBTEXT))
            painter.setFont(QFont("Consolas", 11))
            painter.drawText(self.rect(), Qt.AlignCenter, "데이터 없음")
            return

        start_angle = 90 * 16   # 12시 방향 시작
        legend_y    = int(y + size + 10)

        for label, value, color in self._data:
            span = int(360 * 16 * value / total)
            painter.setBrush(QBrush(QColor(color)))
            painter.setPen(QPen(QColor(C_BG), 2))
            painter.drawPie(rect, start_angle, span)
            start_angle += span

        # 가운데 구멍 (도넛)
        painter.setBrush(QBrush(QColor(C_PANEL)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(hole)

        # 가운데 텍스트
        painter.setPen(QColor(C_TEXT))
        painter.setFont(QFont("Consolas", 18, QFont.Bold))
        painter.drawText(hole.toRect(), Qt.AlignCenter, str(total))
        painter.setFont(QFont("Consolas", 8))
        painter.setPen(QColor(C_SUBTEXT))
        lbl_rect = QRectF(hole.x(), hole.y() + hole.height()*0.55,
                          hole.width(), hole.height()*0.3)
        painter.drawText(lbl_rect, Qt.AlignCenter, "총 처리")

        # 범례
        lx = int(x)
        painter.setFont(QFont("Consolas", 9))
        for i, (label, value, color) in enumerate(self._data):
            ly = legend_y + i * 18
            if ly + 14 > h: break
            painter.setBrush(QBrush(QColor(color)))
            painter.setPen(Qt.NoPen)
            painter.drawRect(lx, ly + 2, 12, 12)
            painter.setPen(QColor(C_TEXT))
            pct = value / total * 100
            painter.drawText(lx + 18, ly + 13, f"{label}  {value}건 ({pct:.0f}%)")


class BarChart(QWidget):
    """시계열 바 차트 위젯"""
    MAX_POINTS = 20

    def __init__(self):
        super().__init__()
        self._history = []   # [(za, zb, nl), ...]
        self._prev    = (0, 0, 0)
        self.setMinimumSize(300, 200)

    def add_snapshot(self, za, zb, nl):
        cur = (za, zb, nl)
        if cur != self._prev:
            self._history.append(cur)
            if len(self._history) > self.MAX_POINTS:
                self._history.pop(0)
            self._prev = cur
            self.update()

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont
        from PyQt5.QtCore import QRectF, Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor(C_PANEL))

        if not self._history:
            painter.setPen(QColor(C_SUBTEXT))
            painter.setFont(QFont("Consolas", 11))
            painter.drawText(self.rect(), Qt.AlignCenter, "데이터 수집 중...")
            return

        pad_l, pad_r, pad_t, pad_b = 40, 20, 20, 40
        chart_w = w - pad_l - pad_r
        chart_h = h - pad_t - pad_b

        max_val = max((za + zb + nl) for za, zb, nl in self._history) or 1

        # 격자선
        painter.setPen(QPen(QColor(C_BORDER), 0.5))
        for i in range(5):
            gy = pad_t + chart_h * i // 4
            painter.drawLine(pad_l, gy, w - pad_r, gy)
            painter.setFont(QFont("Consolas", 8))
            painter.setPen(QColor(C_SUBTEXT))
            val_lbl = str(int(max_val * (4 - i) / 4))
            painter.drawText(2, gy + 4, val_lbl)
            painter.setPen(QPen(QColor(C_BORDER), 0.5))

        # 바
        n      = len(self._history)
        bar_w  = max(4, chart_w // n - 3)
        colors = [C_GREEN, C_PURPLE, C_ORANGE]
        labels = ["A", "B", "없음"]

        for i, (za, zb, nl) in enumerate(self._history):
            bx    = pad_l + i * chart_w // n
            vals  = [za, zb, nl]
            total = sum(vals) or 0
            bottom = pad_t + chart_h

            for val, col in zip(vals, colors):
                if val == 0: continue
                bh = int(chart_h * val / max_val)
                painter.setBrush(QBrush(QColor(col)))
                painter.setPen(Qt.NoPen)
                painter.drawRect(bx, bottom - bh, bar_w, bh)
                bottom -= bh

            # 총합 텍스트
            if total > 0 and bar_w >= 14:
                painter.setPen(QColor(C_SUBTEXT))
                painter.setFont(QFont("Consolas", 7))
                painter.drawText(bx, pad_t + chart_h - int(chart_h * total / max_val) - 3,
                                 str(total))

        # X축
        painter.setPen(QPen(QColor(C_BORDER), 1))
        painter.drawLine(pad_l, pad_t + chart_h, w - pad_r, pad_t + chart_h)

        # 범례
        lx = pad_l
        ly = h - 18
        painter.setFont(QFont("Consolas", 9))
        for lbl, col in [("■ ZONE A", C_GREEN), ("■ ZONE B", C_PURPLE), ("■ 라벨없음", C_ORANGE)]:
            painter.setPen(QColor(col))
            painter.drawText(lx, ly + 12, lbl)
            lx += 100


class LogWindow(QMainWindow):
    """메인 GUI 이벤트 로그를 전체 화면으로 보여주는 독립 창"""
    def __init__(self, parent):
        super().__init__(parent)
        self._p = parent
        self.setWindowTitle("이벤트 로그 — Parcel Control System")
        self.setMinimumSize(860, 600)
        self.setStyleSheet(f"QMainWindow,QWidget{{background:{C_BG};color:{C_TEXT};}}")

        cw = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw); root.setSpacing(8); root.setContentsMargins(14,10,14,10)

        # ── 상단 툴바 ─────────────────────────────────────────
        toolbar = QHBoxLayout()

        title = QLabel("📋  이벤트 로그")
        title.setStyleSheet(f"color:{C_ACCENT}; font-size:16px; font-weight:bold; font-family:Consolas; letter-spacing:2px;")
        toolbar.addWidget(title)
        toolbar.addStretch()

        # 레벨 필터 버튼들
        self._filters = {"ALL":True,"OK":True,"CMD":True,"WARN":True,"ERR":True,"INFO":True,"VISION":True}
        self._filter_btns = {}
        filter_colors = {
            "ALL":   (C_BTN_DARK, C_TEXT),
            "OK":    (C_GREEN,    "#000"),
            "CMD":   (C_ACCENT,   "#000"),
            "WARN":  (C_ORANGE,   "#000"),
            "ERR":   (C_RED,      "#fff"),
            "INFO":  (C_BTN_DARK, C_SUBTEXT),
            "VISION":(C_PURPLE,   "#fff"),
        }
        for lvl, (bg, fg) in filter_colors.items():
            btn = QPushButton(lvl)
            btn.setFixedSize(56, 28)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background:{bg}; color:{fg}; border:none; border-radius:5px;
                    font-family:Consolas; font-size:12px; font-weight:bold;
                }}
                QPushButton:!checked {{
                    background:{C_BTN_DARK}; color:{C_SUBTEXT};
                    border:1px solid {C_BORDER};
                }}
            """)
            btn.toggled.connect(lambda checked, l=lvl: self._toggle_filter(l, checked))
            toolbar.addWidget(btn)
            self._filter_btns[lvl] = btn

        # 검색창
        from PyQt5.QtWidgets import QLineEdit
        self._search = QLineEdit()
        self._search.setPlaceholderText("검색...")
        self._search.setFixedWidth(160)
        self._search.setStyleSheet(f"""
            QLineEdit {{
                background:{C_BTN_DARK}; color:{C_TEXT};
                border:1px solid {C_BORDER}; border-radius:5px;
                font-family:Consolas; font-size:13px; padding:4px 8px;
            }}
        """)
        self._search.textChanged.connect(self._apply_filter)
        toolbar.addWidget(self._search)

        # 버튼들
        btn_clear = QPushButton("🗑  초기화")
        btn_clear.setFixedSize(80, 28)
        btn_clear.setStyleSheet(self._btn_s(C_BTN_DARK, C_SUBTEXT))
        btn_clear.clicked.connect(self._clear)
        toolbar.addWidget(btn_clear)

        btn_export = QPushButton("💾  저장")
        btn_export.setFixedSize(70, 28)
        btn_export.setStyleSheet(self._btn_s(C_BTN_DARK, C_SUBTEXT))
        btn_export.clicked.connect(self._export)
        toolbar.addWidget(btn_export)

        btn_scroll = QPushButton("↓  자동스크롤")
        btn_scroll.setFixedSize(100, 28)
        btn_scroll.setCheckable(True)
        btn_scroll.setChecked(True)
        btn_scroll.setStyleSheet(self._btn_s(C_GREEN, "#000"))
        btn_scroll.toggled.connect(lambda c: btn_scroll.setStyleSheet(
            self._btn_s(C_GREEN,"#000") if c else self._btn_s(C_BTN_DARK,C_SUBTEXT)))
        self._auto_scroll = btn_scroll
        toolbar.addWidget(btn_scroll)

        root.addLayout(toolbar)

        # ── 통계 바 ───────────────────────────────────────────
        stats = QHBoxLayout(); stats.setSpacing(6)
        self._stat_labels = {}
        for lvl, col in [("전체","#E6EDF3"),("OK",C_GREEN),("CMD",C_ACCENT),
                          ("WARN",C_ORANGE),("ERR",C_RED),("INFO",C_SUBTEXT)]:
            w = QWidget()
            w.setStyleSheet(f"QWidget{{background:{C_PANEL};border:1px solid {C_BORDER};border-radius:5px;}}")
            h = QHBoxLayout(w); h.setContentsMargins(8,4,8,4); h.setSpacing(4)
            lbl_k = QLabel(lvl); lbl_k.setStyleSheet(f"color:{C_SUBTEXT};font-size:11px;font-family:Consolas;")
            lbl_v = QLabel("0");  lbl_v.setStyleSheet(f"color:{col};font-size:15px;font-weight:bold;font-family:Consolas;")
            h.addWidget(lbl_k); h.addWidget(lbl_v)
            stats.addWidget(w, stretch=1)
            self._stat_labels[lvl] = lbl_v
        root.addLayout(stats)

        # ── 메인 로그 뷰 ──────────────────────────────────────
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet(f"""
            QTextEdit {{
                background:#080D12; color:{C_SUBTEXT};
                border:1px solid {C_BORDER}; border-radius:8px;
                font-family:Consolas; font-size:13px; padding:8px;
                line-height:1.6;
            }}
            QScrollBar:vertical {{
                background:{C_PANEL}; width:6px; border-radius:3px;
            }}
            QScrollBar::handle:vertical {{
                background:{C_BORDER}; border-radius:3px;
            }}
        """)
        root.addWidget(self._text)

        # ── 하단 상태바 ───────────────────────────────────────
        self._status = QLabel("총 0건")
        self._status.setStyleSheet(f"color:{C_SUBTEXT}; font-size:12px; font-family:Consolas;")
        root.addWidget(self._status)

        # 내부 로그 저장소
        self._entries: list[tuple[str,str,str]] = []  # (ts, level, msg)
        self._counts  = {"전체":0,"OK":0,"CMD":0,"WARN":0,"ERR":0,"INFO":0}

        # 기존 로그 불러오기 (부모의 _log 위젯 HTML)
        self._import_existing()

    def _import_existing(self):
        """부모 GUI의 기존 로그를 불러옴"""
        existing = self._p._log.toPlainText().strip()
        if existing:
            self.append("--- 기존 로그 불러옴 ---", "INFO")
        # 실제로는 부모 로그의 raw HTML을 재파싱하기 어려우므로 안내만 표시

    def append(self, msg: str, level: str = "INFO"):
        """새 로그 항목 추가"""
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._entries.append((ts, level, msg))

        lvl_key = level if level in self._counts else "INFO"
        self._counts["전체"] = self._counts.get("전체", 0) + 1
        self._counts[lvl_key] = self._counts.get(lvl_key, 0) + 1

        # 통계 갱신
        for k, lbl in self._stat_labels.items():
            lbl.setText(str(self._counts.get(k, 0)))
        self._status.setText(f"총 {len(self._entries)}건  |  표시: {self._count_visible()}건")

        # 필터 통과 시 표시
        if self._passes_filter(level, msg):
            self._append_html(ts, level, msg)

    def _passes_filter(self, level, msg):
        lvl_match = self._filters.get("ALL", True) or self._filters.get(level, True)
        search = self._search.text().strip().lower()
        txt_match = (not search) or (search in msg.lower()) or (search in level.lower())
        return self._filters.get(level, True) and txt_match

    def _append_html(self, ts, level, msg):
        colors = {
            "OK":    C_GREEN,   "CMD":   C_ACCENT,
            "WARN":  C_ORANGE,  "ERR":   C_RED,
            "INFO":  C_SUBTEXT, "VISION":C_PURPLE,
        }
        col = colors.get(level, C_SUBTEXT)
        lvl_bg = {"ERR":C_RED,"WARN":C_ORANGE,"OK":C_GREEN}.get(level,"")
        lvl_html = (f'<span style="background:{lvl_bg};color:#000;padding:1px 4px;border-radius:3px;">'
                    f'{level}</span>') if lvl_bg else \
                   f'<span style="color:{col};">[{level}]</span>'
        self._text.append(
            f'<span style="color:{C_BORDER};">{ts}</span> '
            f'{lvl_html} '
            f'<span style="color:{C_TEXT};">{msg}</span>'
        )
        if self._auto_scroll.isChecked():
            sb = self._text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _toggle_filter(self, level, checked):
        self._filters[level] = checked
        self._apply_filter()

    def _apply_filter(self):
        self._text.clear()
        search = self._search.text().strip().lower()
        for ts, level, msg in self._entries:
            if not self._filters.get(level, True):
                continue
            if search and search not in msg.lower() and search not in level.lower():
                continue
            self._append_html(ts, level, msg)
        self._status.setText(f"총 {len(self._entries)}건  |  표시: {self._count_visible()}건")

    def _count_visible(self):
        search = self._search.text().strip().lower()
        cnt = 0
        for _, level, msg in self._entries:
            if not self._filters.get(level, True): continue
            if search and search not in msg.lower() and search not in level.lower(): continue
            cnt += 1
        return cnt

    def _clear(self):
        self._entries.clear()
        self._counts = {"전체":0,"OK":0,"CMD":0,"WARN":0,"ERR":0,"INFO":0}
        for lbl in self._stat_labels.values(): lbl.setText("0")
        self._text.clear()
        self._status.setText("총 0건")
        self.append("로그 초기화됨", "INFO")

    def _export(self):
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SAVE_DIR / f"event_log_{ts}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for ts_, level, msg in self._entries:
                f.write(f"[{ts_}] [{level}] {msg}\n")
        self.append(f"로그 저장됨: {path}", "OK")

    def _btn_s(self, bg, fg):
        return (f"QPushButton{{background:{bg};color:{fg};border:none;border-radius:5px;"
                f"font-family:Consolas;font-size:12px;font-weight:bold;}}"
                f"QPushButton:hover{{opacity:0.85;}}"
                f"QPushButton:!checked{{background:{C_BTN_DARK};color:{C_SUBTEXT};"
                f"border:1px solid {C_BORDER};}}")

    def closeEvent(self, e):
        e.accept()


class DiagWindow(QMainWindow):
    """자동 진단 독립 창 — Hz/BW 실측 + 이상 감지 + 진단 로그"""
    def __init__(self, parent):
        super().__init__(parent)
        self._p = parent
        self.setWindowTitle("시스템 자동 진단 — Parcel Pipeline Monitor")
        self.setMinimumSize(1000, 740)
        self.setStyleSheet(f"QMainWindow,QWidget{{background:{C_BG};color:{C_TEXT};}}")

        cw = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw); root.setSpacing(8); root.setContentsMargins(14,10,14,10)

        # ── 전체 상태 배너 + 자동진단 버튼 ──────────────────────
        top = QHBoxLayout()
        self._overall = QLabel("● 대기 중")
        self._overall.setAlignment(Qt.AlignCenter)
        self._overall.setFixedHeight(38)
        self._overall.setStyleSheet(self._banner_style(C_BTN_DARK, C_SUBTEXT))
        top.addWidget(self._overall, stretch=1)

        self._btn_auto = QPushButton("▶  자동 진단 시작")
        self._btn_auto.setFixedSize(160, 38)
        self._btn_auto.setStyleSheet(self._btn_s(C_ACCENT, "#000"))
        self._btn_auto.clicked.connect(self._toggle_auto_diag)
        top.addWidget(self._btn_auto)

        btn_clear = QPushButton("로그 초기화")
        btn_clear.setFixedSize(90, 38)
        btn_clear.setStyleSheet(self._btn_s(C_BTN_DARK, C_SUBTEXT))
        btn_clear.clicked.connect(self._clear_log)
        top.addWidget(btn_clear)
        root.addLayout(top)

        # ── 파이프라인 다이어그램 ──────────────────────────────
        from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
        from PyQt5.QtGui import QPainter
        self._scene = QGraphicsScene()
        self._view  = QGraphicsView(self._scene)
        self._view.setMinimumHeight(420)
        self._view.setRenderHint(QPainter.Antialiasing)
        self._view.setStyleSheet(
            f"QGraphicsView{{background:#0A0F14;border:1px solid {C_BORDER};border-radius:10px;}}")
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._node_dots = {}
        self._flow_lines = {}
        self._draw_pipeline()
        root.addWidget(self._view)

        # ── 메트릭 카드 ────────────────────────────────────────
        mrow = QHBoxLayout(); mrow.setSpacing(8)
        self._m = {}
        for key, lbl, col in [
            ("cam_fps",  "카메라 FPS",    C_ACCENT),
            ("det_fps",  "감지 FPS",     C_GREEN),
            ("qr_cnt",   "QR 인식 수",   C_PURPLE),
            ("ok_topics","정상 토픽",    C_GREEN),
            ("warn_cnt", "경고",         C_ORANGE),
            ("err_cnt",  "오류",         C_RED),
        ]:
            self._m[key] = self._metric_card(mrow, lbl, "—", col)
        root.addLayout(mrow)

        # ── 토픽 테이블 + 진단 로그 (좌우 분할) ──────────────
        split = QHBoxLayout(); split.setSpacing(8)

        # 토픽 테이블
        tbl_grp = self._grp("토픽 상세 상태  (Hz · BW · 판정)")
        tg = QVBoxLayout(tbl_grp); tg.setSpacing(2); tg.setContentsMargins(8,8,8,6)
        hdr = QWidget(); hdr.setStyleSheet(f"background:{C_BTN_DARK};border-radius:4px;")
        hh  = QHBoxLayout(hdr); hh.setContentsMargins(10,4,10,4); hh.setSpacing(0)
        for txt, w_, align in [
            ("토픽",  200, Qt.AlignLeft|Qt.AlignVCenter),
            ("노드",  120, Qt.AlignLeft|Qt.AlignVCenter),
            ("Hz",     56, Qt.AlignRight|Qt.AlignVCenter),
            ("BW",     64, Qt.AlignRight|Qt.AlignVCenter),
            ("경과",   56, Qt.AlignRight|Qt.AlignVCenter),
            ("판정",   52, Qt.AlignRight|Qt.AlignVCenter),
        ]:
            l = QLabel(txt); l.setFixedWidth(w_); l.setAlignment(align)
            l.setStyleSheet(f"color:{C_SUBTEXT};font-size:11px;font-family:Consolas;font-weight:bold;")
            hh.addWidget(l)
        tg.addWidget(hdr)
        self._rows = {}
        for topic, node, col in [
            ("/rgb/compressed",              "image_transport", C_ACCENT),
            ("/parcel_detections/annotated", "parcel_detector", C_GREEN),
            ("/parcel_with_qr",              "parcel_detector", C_GREEN),
            ("/parcel_no_label",             "parcel_detector", C_GREEN),
            ("/qr_crop_image",               "qr_decoder_node", C_PURPLE),
            ("/qr_code",                     "qr_decoder_node", C_PURPLE),
        ]:
            self._rows[topic] = self._add_row(tg, topic, node, col)
        split.addWidget(tbl_grp, stretch=3)

        # 자동 진단 로그
        log_grp = self._grp("자동 진단 로그")
        lg = QVBoxLayout(log_grp); lg.setContentsMargins(6,8,6,6)
        self._log_widget = QTextEdit()
        self._log_widget.setReadOnly(True)
        self._log_widget.setStyleSheet(f"""
            QTextEdit{{background:#0D1117;color:{C_SUBTEXT};
              border:none;border-radius:5px;
              font-family:Consolas;font-size:12px;padding:4px;}}
        """)
        lg.addWidget(self._log_widget)
        split.addWidget(log_grp, stretch=2)
        root.addLayout(split)

        # ── 허브 피드백 상태 바 ────────────────────────────────
        from PyQt5.QtWidgets import QLabel as _QLabel
        self._hub_info_lbl = _QLabel("허브 피드백 대기 중...")
        self._hub_info_lbl.setStyleSheet(f"""
            QLabel {{
                background:{C_BTN_DARK}; color:{C_SUBTEXT};
                border:1px solid {C_BORDER}; border-radius:5px;
                font-family:Consolas; font-size:11px; padding:5px 10px;
            }}
        """)
        self._hub_info_lbl.setTextFormat(Qt.RichText)
        root.addWidget(self._hub_info_lbl)

        # ── 타이머 ────────────────────────────────────────────
        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self._refresh)
        self._refresh_timer.start(1000)

        self._auto_diag_timer = QTimer()
        self._auto_diag_timer.timeout.connect(self._run_auto_diag)
        self._auto_running = False

        # Hz/BW 1초 윈도우 리셋 타이머
        self._bw_timer = QTimer()
        self._bw_timer.timeout.connect(self._calc_hz_bw)
        self._bw_timer.start(1000)

        self._hz_snapshot  = {k: 0.0 for k in self._p._topic_last_recv}
        self._bw_snapshot  = {k: 0.0 for k in self._p._topic_last_recv}

        self._refresh()
        self._diag_log("시스템", "INFO", "진단 창 열림. ▶ 자동 진단 시작 버튼으로 자동 점검 시작.")

    # ── 파이프라인 다이어그램 ──────────────────────────────────
    def _draw_pipeline(self):
        from PyQt5.QtGui import QPen, QBrush, QColor, QPainterPath, QFont
        from PyQt5.QtCore import QRectF, Qt as Qt_
        import math
        sc = self._scene; sc.clear()
        W,H = 960,410; sc.setSceneRect(0,0,W,H)

        def nd(x,y,w,h,title,sub="",bc="#30363D"):
            sc.addRect(QRectF(x,y,w,h),QPen(QColor(bc),1.2),QBrush(QColor("#161B22")))
            t=sc.addText(title); t.setDefaultTextColor(QColor("#E6EDF3"))
            f=QFont("Consolas",9); f.setBold(True); t.setFont(f)
            t.setPos(x+w/2-t.boundingRect().width()/2,y+10)
            if sub:
                s=sc.addText(sub); s.setDefaultTextColor(QColor("#8B949E"))
                fs=QFont("Consolas",8); s.setFont(fs)
                s.setPos(x+w/2-s.boundingRect().width()/2,y+30)

        def arr(x1,y1,x2,y2,col,dash=False):
            pen=QPen(QColor(col),1.8)
            if dash: pen.setStyle(Qt_.DashLine)
            ln=sc.addLine(x1,y1,x2,y2,pen)
            dx,dy=x2-x1,y2-y1; ang=math.atan2(dy,dx); sz=9
            p=QPainterPath()
            p.moveTo(x2-sz*math.cos(ang-0.4),y2-sz*math.sin(ang-0.4))
            p.lineTo(x2,y2)
            p.lineTo(x2-sz*math.cos(ang+0.4),y2-sz*math.sin(ang+0.4))
            sc.addPath(p,QPen(QColor(col),1.8),QBrush(Qt_.NoBrush))
            return ln

        def dot(x,y,col="#8B949E"):
            d=sc.addEllipse(QRectF(x-7,y-7,14,14),QPen(Qt_.NoPen),QBrush(QColor(col)))
            return d

        def lbl(x,y,txt,col,size=8):
            t=sc.addText(txt); t.setDefaultTextColor(QColor(col))
            f=QFont("Consolas",size); t.setFont(f); t.setPos(x,y)

        # ── PC 영역 ───────────────────────────────────────────
        sc.addRect(QRectF(6,18,172,350),QPen(QColor("#30363D"),0.6,Qt_.DashLine),QBrush(Qt_.NoBrush))
        lbl(38,8,"IsaacSim05","#8B949E",8)
        sc.addRect(QRectF(188,8,762,368),QPen(QColor("#30363D"),0.6,Qt_.DashLine),QBrush(Qt_.NoBrush))
        lbl(500,1,"Vision PC  (taehwan)","#8B949E",8)

        # ── Isaac Sim PC ──────────────────────────────────────
        nd(12,32,158,60,"Isaac Sim","시뮬레이션")
        nd(12,140,158,60,"RealSense D455","/rgb  raw")
        self._node_dots["isaac_sim"] = dot(166,170)

        # ── image_transport ───────────────────────────────────
        nd(196,108,160,64,"image_transport","raw→/rgb/compressed","#00D9FF")
        self._node_dots["image_transport"] = dot(352,140)

        # ── parcel_hub_node (중앙 허브) ───────────────────────
        nd(374,80,168,80,"parcel_hub_node","중앙 허브 · 워치독","#E3B341")
        lbl(380,128,"/hub/rgb/compressed","#E3B341",7)
        self._node_dots["parcel_hub"] = dot(538,120)

        # ── parcel_detector ───────────────────────────────────
        nd(558,28,170,70,"parcel_detector","YOLO11n  mAP50=0.995","#3FB950")
        self._node_dots["parcel_detector"] = dot(724,63)

        # ── qr_decoder ────────────────────────────────────────
        nd(558,170,170,66,"qr_decoder_node","pyzbar + threading","#A371F7")
        self._node_dots["qr_decoder_node"] = dot(724,203)

        # ── Control GUI ───────────────────────────────────────
        nd(746,18,200,260,"Control GUI","parcel_gui_node","#00D9FF")
        lbl(754,56, "/rgb/compressed",       "#00D9FF",7)
        lbl(754,70, "/hub/state",            "#E3B341",7)
        lbl(754,84, "/hub/alert",            "#E3B341",7)
        lbl(754,100,"/parcel_detections",    "#3FB950",7)
        lbl(754,114,"/parcel_with_qr",       "#3FB950",7)
        lbl(754,128,"/qr_code",              "#A371F7",7)
        lbl(754,142,"/qr_crop_image",        "#A371F7",7)
        sc.addLine(754,162,940,162,QPen(QColor("#30363D"),0.5))
        lbl(754,168,"→ /simulation_control", "#E3B341",7)
        lbl(754,184,"→ /emergency_stop",     "#F85149",7)
        lbl(754,198,"→ /cmd/detection_enable","#00D9FF",7)
        lbl(754,212,"→ /cmd/conf_threshold", "#00D9FF",7)
        lbl(754,226,"→ /cmd/relay_enable",   "#E3B341",7)

        # ── 화살표 ────────────────────────────────────────────
        # Isaac → image_transport
        self._flow_lines["rgb"]       = arr(170,170, 194,140, "#00D9FF")
        # image_transport → hub
        self._flow_lines["to_hub"]    = arr(356,130, 372,120, "#E3B341")
        # hub → parcel_detector
        self._flow_lines["hub_det"]   = arr(542,100, 556,58,  "#E3B341")
        # hub → qr_decoder
        self._flow_lines["hub_qr"]    = arr(542,140, 556,192, "#E3B341")
        # detector → GUI
        self._flow_lines["det_gui"]   = arr(728,63,  744,90,  "#3FB950")
        # qr_decoder → GUI
        self._flow_lines["qr_gui"]    = arr(728,203, 744,145, "#A371F7")
        # hub → GUI (상태/경고) — 허브 우측 중앙에서 GUI 좌측으로 직선
        self._flow_lines["hub_gui"]   = arr(542,120, 744,148, "#E3B341")

        # ── 워치독 피드백 (detector/qr → hub, 점선) ──────────
        pen_fb=QPen(QColor("#30363D"),1.0); pen_fb.setStyle(Qt_.DashLine)
        # detector → hub 피드백
        sc.addLine(643,63,  460,63,  pen_fb)
        sc.addLine(460,63,  460,100, pen_fb)
        p1=QPainterPath(); ang=math.atan2(1,0); sz=6
        p1.moveTo(460-sz*math.cos(ang-0.4),100-sz*math.sin(ang-0.4))
        p1.lineTo(460,100); p1.lineTo(460-sz*math.cos(ang+0.4),100-sz*math.sin(ang+0.4))
        sc.addPath(p1,QPen(QColor("#30363D"),1.0),QBrush(Qt_.NoBrush))
        lbl(465,58,"/state/detector","#484F58",7)
        # qr_decoder → hub 피드백
        sc.addLine(643,200, 475,200, pen_fb)
        sc.addLine(475,200, 475,140, pen_fb)
        p3=QPainterPath(); ang2=math.atan2(-1,0); sz2=6
        p3.moveTo(475-sz2*math.cos(ang2-0.4),140-sz2*math.sin(ang2-0.4))
        p3.lineTo(475,140); p3.lineTo(475-sz2*math.cos(ang2+0.4),140-sz2*math.sin(ang2+0.4))
        sc.addPath(p3,QPen(QColor("#30363D"),1.0),QBrush(Qt_.NoBrush))
        lbl(465,184,"/state/qr_decoder","#484F58",7)

        # ── GUI → Isaac 제어 역방향 ───────────────────────────
        pen_c=QPen(QColor("#E3B341"),1.2); pen_c.setStyle(Qt_.DotLine)
        sc.addLine(744,245, 90,245, pen_c)
        sc.addLine(90,245,  90,200, pen_c)
        p2=QPainterPath(); ang3=math.atan2(-1,0); sz3=8
        p2.moveTo(90-sz3*math.cos(ang3-0.4),200-sz3*math.sin(ang3-0.4))
        p2.lineTo(90,200); p2.lineTo(90-sz3*math.cos(ang3+0.4),200-sz3*math.sin(ang3+0.4))
        sc.addPath(p2,QPen(QColor("#E3B341"),1.2),QBrush(Qt_.NoBrush))

        # ── 범례 ─────────────────────────────────────────────
        lbl(196,385,"━ 카메라","#00D9FF",8)
        lbl(316,385,"━ 허브배포","#E3B341",8)
        lbl(436,385,"━ 감지결과","#3FB950",8)
        lbl(556,385,"━ QR","#A371F7",8)
        lbl(636,385,"-- 피드백","#484F58",8)
        lbl(730,385,"··· 제어","#E3B341",8)

    # ── Hz/BW 1초 스냅샷 ─────────────────────────────────────
    def _calc_hz_bw(self):
        p = self._p
        for topic in p._topic_recv_cnt:
            win = p._topic_hz_window.get(topic, [])
            if win:
                avg_interval = sum(win) / len(win)
                self._hz_snapshot[topic] = round(1.0 / avg_interval, 1) if avg_interval > 0 else 0.0
            else:
                self._hz_snapshot[topic] = 0.0
            self._bw_snapshot[topic] = p._topic_recv_bw.get(topic, 0) / 1024  # KB/s
            p._topic_recv_bw[topic] = 0  # 1초 리셋

    # ── 자동 진단 토글 ────────────────────────────────────────
    def _toggle_auto_diag(self):
        if self._auto_running:
            self._auto_diag_timer.stop()
            self._auto_running = False
            self._btn_auto.setText("▶  자동 진단 시작")
            self._btn_auto.setStyleSheet(self._btn_s(C_ACCENT, "#000"))
            self._diag_log("자동진단", "INFO", "자동 진단 중지됨")
        else:
            self._auto_diag_timer.start(3000)  # 3초마다 진단
            self._auto_running = True
            self._btn_auto.setText("⬛  자동 진단 중지")
            self._btn_auto.setStyleSheet(self._btn_s(C_RED, "#fff"))
            self._diag_log("자동진단", "OK", "자동 진단 시작 — 3초 간격으로 점검")
            self._run_auto_diag()  # 즉시 1회 실행

    def _run_auto_diag(self):
        """자동 진단 — Hz/BW/타임아웃 종합 점검"""
        p = self._p
        if p._state != "running":
            self._diag_log("자동진단", "WARN", "시스템이 실행 중이 아닙니다")
            return

        now = time.time()
        issues = []

        for topic, (hz_min, hz_max) in p._topic_hz_range.items():
            hz  = self._hz_snapshot.get(topic, 0.0)
            bw  = self._bw_snapshot.get(topic, 0.0)
            last = p._topic_last_recv.get(topic)
            timeout = p._topic_timeout.get(topic, 3.0)

            # 타임아웃 체크
            if last is None:
                issues.append(("ERR", f"{topic} — 수신 없음 (한 번도 수신 못함)"))
                continue
            elapsed = now - last
            if elapsed > timeout * 2:
                issues.append(("ERR", f"{topic} — 타임아웃 {elapsed:.1f}s (기준 {timeout}s)"))
                continue

            # Hz 이상 체크 (토픽이 활성 상태일 때만)
            if hz > 0:
                if hz < hz_min:
                    issues.append(("WARN", f"{topic} — Hz 낮음 {hz:.1f}Hz (기준 ≥{hz_min}Hz)"))
                elif hz > hz_max:
                    issues.append(("WARN", f"{topic} — Hz 과다 {hz:.1f}Hz (기준 ≤{hz_max}Hz)"))

            # BW 이상 체크 (이미지 토픽만 — 10MB/s 초과 시 경고)
            if "compressed" in topic or "image" in topic or "annotated" in topic:
                if bw > 10240:  # 10MB/s
                    issues.append(("WARN", f"{topic} — BW 높음 {bw/1024:.1f}MB/s"))

        if not issues:
            self._diag_log("자동진단", "OK", f"전체 점검 완료 — 이상 없음 (토픽 {len(p._topic_last_recv)}개 정상)")
        else:
            for level, msg in issues:
                self._diag_log("자동진단", level, msg)

    # ── 갱신 ─────────────────────────────────────────────────
    def _refresh(self):
        from PyQt5.QtGui import QBrush, QColor
        p   = self._p
        now = time.time()
        all_ok = True; any_err = False
        ok_cnt = 0; warn_cnt = 0; err_cnt = 0

        topic_to_node = {
            "/rgb/compressed":              "image_transport",
            "/parcel_detections/annotated": "parcel_detector",
            "/parcel_with_qr":              "parcel_detector",
            "/parcel_no_label":             "parcel_detector",
            "/qr_crop_image":               "qr_decoder_node",
            "/qr_code":                     "qr_decoder_node",
        }
        node_ok = {k: True for k in self._node_dots}
        node_ok["isaac_sim"] = (p._state == "running")

        # ── /hub/state 피드백으로 노드 상태 결정 ─────────────
        hub_data  = getattr(p, '_hub_state', {})
        hub_nodes = hub_data.get('nodes', {}) if hub_data else {}

        if hub_data:
            # 허브 자체 (relay_enabled)
            relay_ok = hub_data.get('relay_enabled', False)
            node_ok["parcel_hub"] = relay_ok

            # parcel_detector — /state/detector 피드백 기반
            det_info   = hub_nodes.get('parcel_detector', {})
            det_status = det_info.get('status', 'waiting')   # ok/timeout/waiting
            det_data   = det_info.get('data', {})
            det_fps    = det_data.get('fps', 0.0)
            det_enabled= det_data.get('enabled', True)
            det_conf   = det_data.get('conf', 0.5)
            det_retry  = det_info.get('retry', 0)
            node_ok["parcel_detector"] = (det_status == 'ok' and det_enabled)

            # qr_decoder — /state/qr_decoder 피드백 기반
            qr_info    = hub_nodes.get('qr_decoder', {})
            qr_status  = qr_info.get('status', 'waiting')
            qr_data    = qr_info.get('data', {})
            qr_rate    = qr_data.get('success_rate', 0.0)
            qr_enabled = qr_data.get('enabled', True)
            qr_retry   = qr_info.get('retry', 0)
            node_ok["qr_decoder_node"] = (qr_status == 'ok' and qr_enabled)

            # 다이어그램 노드 툴팁 정보 업데이트 (토픽 테이블 하단에 표시)
            self._update_hub_info(
                relay_ok, det_status, det_fps, det_enabled, det_conf, det_retry,
                qr_status, qr_rate, qr_enabled, qr_retry
            )

        for topic, row in self._rows.items():
            last    = p._topic_last_recv.get(topic)
            timeout = p._topic_timeout.get(topic, 3.0)
            hz      = self._hz_snapshot.get(topic, 0.0)
            bw      = self._bw_snapshot.get(topic, 0.0)
            hz_min, hz_max = p._topic_hz_range.get(topic, (0,999))

            if p._state != "running":
                status = "idle"
            elif last is None:
                status = "error"
            else:
                es = now - last
                if es > timeout * 2:   status = "error"
                elif es > timeout:     status = "warn"
                elif hz > 0 and (hz < hz_min or hz > hz_max): status = "warn"
                else:                  status = "ok"

            if status == "ok":
                col = C_GREEN;   ok_cnt += 1
            elif status == "warn":
                col = C_ORANGE;  warn_cnt += 1; all_ok = False
            elif status == "error":
                col = C_RED;     err_cnt += 1; all_ok = False; any_err = True
                nd = topic_to_node.get(topic)
                if nd: node_ok[nd] = False
            else:
                col = C_SUBTEXT

            hz_txt  = f"{hz:.1f}" if hz > 0 else "—"
            bw_txt  = f"{bw:.0f}K" if bw < 1024 else f"{bw/1024:.1f}M"
            el_txt  = f"{now-last:.1f}s" if last else "—"
            st_txt  = {"ok":"OK","warn":"WARN","error":"ERR","idle":"—"}.get(status,"—")

            row["dot"].setStyleSheet(f"color:{col};font-size:15px;")
            for key, txt in [("hz",hz_txt),("bw",bw_txt),("elapsed",el_txt),("status",st_txt)]:
                row[key].setText(txt)
                row[key].setStyleSheet(f"color:{col};font-size:12px;font-family:Consolas;"
                                       + ("font-weight:bold;" if key=="status" else ""))

        # 다이어그램 점 색상 업데이트
        from PyQt5.QtGui import QBrush, QColor
        for nid, dot_item in self._node_dots.items():
            if p._state != "running":
                c = C_SUBTEXT
            elif not node_ok.get(nid, True):
                c = C_RED
            elif nid in ("parcel_hub","parcel_detector","qr_decoder_node") and not hub_data:
                c = C_ORANGE   # 허브 피드백 아직 없음
            else:
                c = C_GREEN
            dot_item.setBrush(QBrush(QColor(c)))

        # 메트릭
        fps = p._cam_conv.get_fps()
        self._m["cam_fps"].setText(f"{fps:.0f}" if fps>0 else "—")
        self._m["qr_cnt"].setText(str(p._qr_count))
        self._m["ok_topics"].setText(f"{ok_cnt}/6")
        self._m["warn_cnt"].setText(str(warn_cnt))
        self._m["err_cnt"].setText(str(err_cnt))
        ok_col = C_GREEN if ok_cnt==6 else (C_ORANGE if ok_cnt>=3 else C_RED)
        self._m["ok_topics"].setStyleSheet(f"color:{ok_col};font-size:22px;font-weight:bold;font-family:Consolas;")
        err_col = C_RED if err_cnt>0 else C_SUBTEXT
        self._m["err_cnt"].setStyleSheet(f"color:{err_col};font-size:22px;font-weight:bold;font-family:Consolas;")

        # 배너
        if p._state != "running":
            txt="●  시스템 미실행"; bg=C_BTN_DARK; fg=C_SUBTEXT
        elif any_err:
            txt=f"✖  오류 {err_cnt}건 — 즉시 확인 필요"; bg=C_RED; fg="#fff"
        elif not all_ok:
            txt=f"⚠  경고 {warn_cnt}건 — 토픽 상태 점검 권장"; bg=C_ORANGE; fg="#000"
        else:
            txt="✔  모든 시스템 정상"; bg=C_GREEN; fg="#000"
        self._overall.setText(txt)
        self._overall.setStyleSheet(self._banner_style(bg, fg))

    # ── 진단 로그 ─────────────────────────────────────────────
    def refresh_from_hub(self, hub_data: dict):
        """GUI의 _on_hub_state에서 호출 — 허브 수신 즉시 다이어그램 갱신"""
        # _refresh()가 1초 타이머로도 돌고 있으니 여기서는 다이어그램 점만 즉시 갱신
        from PyQt5.QtGui import QBrush, QColor
        p         = self._p
        hub_nodes = hub_data.get('nodes', {})
        relay_ok  = hub_data.get('relay_enabled', False)

        node_map = {
            "parcel_hub":      relay_ok,
            "parcel_detector": hub_nodes.get('parcel_detector', {}).get('status') == 'ok',
            "qr_decoder_node": hub_nodes.get('qr_decoder',     {}).get('status') == 'ok',
        }
        for nid, ok in node_map.items():
            dot = self._node_dots.get(nid)
            if dot:
                c = C_GREEN if ok else C_RED
                if p._state != "running": c = C_SUBTEXT
                dot.setBrush(QBrush(QColor(c)))

    def _update_hub_info(self, relay_ok,
                         det_status, det_fps, det_enabled, det_conf, det_retry,
                         qr_status,  qr_rate, qr_enabled,  qr_retry):
        """허브 피드백 상세 정보를 하단 info 레이블에 표시"""
        if not hasattr(self, '_hub_info_lbl'):
            return
        det_col = C_GREEN if det_status == 'ok' else (C_ORANGE if det_status == 'waiting' else C_RED)
        qr_col  = C_GREEN if qr_status  == 'ok' else (C_ORANGE if qr_status  == 'waiting' else C_RED)
        relay_col = C_GREEN if relay_ok else C_RED

        self._hub_info_lbl.setText(
            f'<span style="color:{relay_col}">허브 재배포: {"ON" if relay_ok else "OFF"}</span>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;'
            f'<span style="color:{det_col}">detector: {det_status}'
            f'  fps={det_fps:.1f}  conf={det_conf:.2f}'
            f'  {"ON" if det_enabled else "OFF"}'
            f'  재시도={det_retry}</span>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;'
            f'<span style="color:{qr_col}">qr_decoder: {qr_status}'
            f'  성공률={qr_rate:.0f}%'
            f'  {"ON" if qr_enabled else "OFF"}'
            f'  재시도={qr_retry}</span>'
        )

    def _diag_log(self, src: str, level: str, msg: str):
        colors = {"INFO":C_SUBTEXT,"OK":C_GREEN,"WARN":C_ORANGE,"ERR":C_RED}
        col = colors.get(level, C_SUBTEXT)
        ts  = datetime.datetime.now().strftime("%H:%M:%S")
        self._log_widget.append(
            f'<span style="color:{C_BORDER}">[{ts}]</span> '
            f'<span style="color:{col}">[{level}]</span> '
            f'<span style="color:{C_SUBTEXT}">{src}</span> '
            f'<span style="color:{C_TEXT}">{msg}</span>'
        )
        sb = self._log_widget.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _clear_log(self):
        self._log_widget.clear()
        self._diag_log("시스템", "INFO", "로그 초기화됨")

    # ── 헬퍼 위젯 ─────────────────────────────────────────────
    def _add_row(self, layout, topic, node, color):
        row_w = QWidget()
        row_w.setStyleSheet("QWidget:hover{background:#1C2128;}")
        rh = QHBoxLayout(row_w); rh.setContentsMargins(10,5,10,5); rh.setSpacing(0)
        dot = QLabel("●"); dot.setFixedWidth(16)
        dot.setStyleSheet(f"color:{C_SUBTEXT};font-size:13px;")
        # dot(16) + topic명(184) = 200 → 헤더 "토픽" 200과 일치
        nm = QLabel(topic); nm.setFixedWidth(184)
        nm.setStyleSheet(f"color:{color};font-size:11px;font-family:Consolas;font-weight:bold;")
        nd = QLabel(node); nd.setFixedWidth(120)
        nd.setStyleSheet(f"color:{C_SUBTEXT};font-size:11px;font-family:Consolas;")
        hz = QLabel("—"); hz.setFixedWidth(56); hz.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        bw = QLabel("—"); bw.setFixedWidth(64); bw.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        el = QLabel("—"); el.setFixedWidth(56); el.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        st = QLabel("—"); st.setFixedWidth(52); st.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        for w_ in [dot,nm,nd,hz,bw,el,st]: rh.addWidget(w_)
        layout.addWidget(row_w)
        return {"dot":dot,"hz":hz,"bw":bw,"elapsed":el,"status":st}

    def _metric_card(self, layout, label, val, color):
        card = QWidget()
        card.setStyleSheet(
            f"QWidget{{background:{C_PANEL};border:1px solid {C_BORDER};border-radius:8px;}}")
        v = QVBoxLayout(card); v.setContentsMargins(12,8,12,8); v.setSpacing(2)
        vl = QLabel(val)
        vl.setStyleSheet(f"color:{color};font-size:22px;font-weight:bold;font-family:Consolas;")
        nl = QLabel(label)
        nl.setStyleSheet(f"color:{C_SUBTEXT};font-size:11px;font-family:Consolas;")
        v.addWidget(vl); v.addWidget(nl)
        layout.addWidget(card, stretch=1)
        return vl

    def _grp(self, title):
        g = QGroupBox(title)
        g.setStyleSheet(f"""
            QGroupBox{{background:#1C2128;border:1px solid {C_BORDER};border-radius:8px;
              color:{C_SUBTEXT};font-size:12px;font-family:Consolas;
              margin-top:8px;padding-top:6px;}}
            QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 4px;}}
        """)
        return g

    def _banner_style(self, bg, fg):
        return (f"QLabel{{background:{bg};color:{fg};border-radius:7px;"
                f"font-family:Consolas;font-size:15px;font-weight:bold;letter-spacing:2px;}}")

    def _btn_s(self, bg, fg):
        return (f"QPushButton{{background:{bg};color:{fg};border:none;border-radius:6px;"
                f"font-family:Consolas;font-size:13px;font-weight:bold;}}"
                f"QPushButton:hover{{opacity:0.85;}}")

    def closeEvent(self, e):
        self._refresh_timer.stop()
        self._auto_diag_timer.stop()
        self._bw_timer.stop()
        e.accept()


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Parcel Central Controller")
    win = ParcelControlGUI()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

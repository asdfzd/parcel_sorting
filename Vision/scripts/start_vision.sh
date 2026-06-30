#!/bin/bash

# ─────────────────────────────────────────
# Vision Pipeline Startup Script (tmux)
# ─────────────────────────────────────────
# 패널 구성:
#   [패널 0] image_transport     /rgb → /rgb/compressed
#   [패널 1] parcel_hub_node     중앙 허브 (영상 재배포 + 워치독)
#   [패널 2] parcel_detector     YOLO11 감지
#   [패널 3] qr_decoder_node     QR 디코딩
#   [패널 4] parcel_control_gui  중앙 제어 GUI
# ─────────────────────────────────────────

SESSION="vision"

# ── 경로 동적 탐색 (워크스페이스 이름 무관) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"        # .../src/cobot3
SRC_DIR="$(dirname "$PKG_DIR")"           # .../src
WS_DIR="$(dirname "$SRC_DIR")"            # 워크스페이스 루트

MODEL_PATH="$PKG_DIR/models/parcel_qr_det.pt"
WS_SETUP="source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash && export ROS_DOMAIN_ID=103"

# ── 사전 확인 ─────────────────────────────
if [ ! -f "$MODEL_PATH" ]; then
  echo "❌ 모델 파일 없음: $MODEL_PATH"
  exit 1
fi

if [ ! -f "$WS_DIR/install/setup.bash" ]; then
  echo "❌ 워크스페이스 빌드 필요: cd $WS_DIR && colcon build"
  exit 1
fi

# 기존 세션 있으면 종료
tmux kill-session -t $SESSION 2>/dev/null
sleep 0.5

# 새 세션 생성 (백그라운드)
tmux new-session -d -s $SESSION -x 240 -y 60

# ── 패널 0: image_transport (좌측) ───────
tmux send-keys -t $SESSION \
  "$WS_SETUP && echo '🔄 [1/5] image_transport republisher' && \
  ros2 run image_transport republish raw \
  --ros-args \
  --remap in:=/rgb \
  --remap out/compressed:=/rgb/compressed" Enter

# 우측으로 분할
tmux split-window -h -t $SESSION
sleep 0.3

# ── 패널 1: parcel_hub_node (우측 상단) ──
tmux send-keys -t $SESSION \
  "$WS_SETUP && sleep 2 && echo '🔀 [2/5] parcel_hub_node (중앙 허브)' && \
  ros2 run cobot3 parcel_hub_node \
  --ros-args \
  -p input_topic:=/rgb/compressed \
  -p output_topic:=/hub/rgb/compressed \
  -p enable_watchdog:=true" Enter

tmux split-window -v -t $SESSION
sleep 0.3

# ── 패널 2: parcel_detector (우측 두번째) ─
tmux send-keys -t $SESSION \
  "$WS_SETUP && sleep 4 && echo '🎯 [3/5] parcel_detector' && \
  ros2 launch cobot3 parcel_detector.launch.py \
  model_path:=$MODEL_PATH \
  rgb_topic:=/hub/rgb/compressed" Enter

tmux split-window -v -t $SESSION
sleep 0.3

# ── 패널 3: qr_decoder (우측 세번째) ─────
tmux send-keys -t $SESSION \
  "$WS_SETUP && sleep 6 && echo '📦 [4/5] qr_decoder_node' && \
  ros2 run cobot3 qr_decoder_node \
  --ros-args \
  -p rgb_topic:=/hub/rgb/compressed \
  -p publish_only_on_change:=false" Enter

tmux split-window -v -t $SESSION
sleep 0.3

# ── 패널 4: GUI (우측 하단) ──────────────
tmux send-keys -t $SESSION \
  "$WS_SETUP && sleep 10 && echo '🖥️  [5/5] parcel_control_gui' && \
  ros2 run cobot3 parcel_control_gui" Enter

# 레이아웃 정리
tmux select-layout -t $SESSION main-vertical
tmux select-pane -t $SESSION:0.0

# ── 안내 출력 ─────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     Parcel Vision Pipeline 시작           ║"
echo "╠══════════════════════════════════════════╣"
echo "║  [0] image_transport   /rgb → /rgb/compressed"
echo "║  [1] parcel_hub_node   중앙 허브 (워치독)  ║"
echo "║  [2] parcel_detector   YOLO11 감지        ║"
echo "║  [3] qr_decoder_node   QR 디코딩          ║"
echo "║  [4] parcel_control_gui GUI               ║"
echo "╠══════════════════════════════════════════╣"
echo "║  워크스페이스: $WS_DIR"
echo "║  Isaac Sim 는 별도 터미널에서 먼저 실행   ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  패널 전환:  Ctrl+B → 화살표키"
echo "  세션 나가기: Ctrl+B → D"
echo ""

tmux attach -t $SESSION

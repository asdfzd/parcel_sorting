# 🤖 cobot3 — 자동 택배 분류 및 검사 시스템 (Isaac Sim + ROS2 + AI Vision)

`cobot3`는 ROS2 Humble 기반의 택배 박스 비전 인식 패키지입니다. Isaac Sim(또는 일반 ROS2 카메라 토픽)에서 입력되는 영상 스트림을 받아 **YOLO11** 모델로 택배 박스(`package`)와 QR 라벨(`qr_label`)을 감지하고, QR 영역을 crop하여 배송 구역(ZONE) 정보를 판독한 뒤 **PyQt5 GUI**에서 시스템 상태와 인식 결과를 모니터링할 수 있도록 구성되어 있습니다.

프로젝트의 최종 목표는 컨베이어 위로 이동하는 택배 박스를 인식하고, QR 또는 송장 정보를 기준으로 `ZONE_A` ~ `ZONE_E`와 같은 목적지 구역을 판단한 뒤, **Doosan M0609 로봇팔 + VGC10 흡착 그리퍼**를 이용해 박스를 지정 위치로 분류하는 것입니다.

현재 `cobot3` 패키지는 전체 시스템 중 **비전 인식 · QR 디코딩 · 상태 모니터링 GUI**를 담당합니다. M0609 로봇팔 제어, VGC10 흡착 그리퍼, Isaac Sim USD 환경 구성은 별도 로봇팔 실행 파트에서 담당하며, 두 파트는 ROS2 토픽(`/qr_code`)을 통해 연동됩니다.

> ⚠️ 본 문서는 제출 가이드라인 기준 임시(draft) 통합본입니다. 본문 곳곳의 "확인 필요" 표시 항목은 최종 제출 전 검증 및 보완이 필요합니다.

---

## 📌 주요 기능 (Key Features)

### 1. 택배 / QR 인식 (Parcel & QR Detection)
- **탐지:** YOLO11 모델로 컨베이어 위 택배(`package`)와 QR 라벨(`qr_label`) bbox를 각각 탐지하고, 가장 가까운 쌍끼리 매칭합니다.
- **디코딩:** 매칭된 QR 라벨 bbox를 crop한 뒤 `pyzbar`로 우선 디코딩하고, 실패 시 OpenCV `QRCodeDetector`로 fallback합니다. 디코딩 성공 시 결과를 `/qr_code`로 발행합니다. (전체 이미지 직접 스캔 방식은 위치 불확실성·해상도 문제로 폐기, bbox crop 방식으로 전환)
- **정확도:** Isaac Sim Replicator로 생성한 합성 데이터 300장으로 학습, **mAP50 = 0.995** 달성.
- QR 라벨이 없는 박스는 `/parcel_no_label`로 예외 처리됩니다.

### 2. 분류 및 게이트 제어 (Sorting & Gate Control)
- `/qr_code`로 발행된 ZONE 값(`ZONE_A` ~ `ZONE_E`)에 따라 택배를 해당 구역으로 분류합니다.
- 게이트 제어는 별도 `/simulation_control` 토픽이 아닌, **컨베이어 커터(물리적 게이트)** 를 통해 수행합니다.
- `parcel_hub_node`가 중앙에서 영상 재배포, 노드 상태 모니터링, 워치독(타임아웃 시 재명령)을 관리하며 `/hub/state`를 퍼블리시합니다.

### 3. 실시간 모니터링 GUI (PyQt5)
- 컨베이어 영상, YOLO 감지 시각화, QR crop 이미지, QR 판독 결과를 한 화면에서 확인합니다.
- 허브 · detector · QR decoder 상태를 모니터링하고, emergency stop / reset / confidence threshold 조절 등의 제어 UI를 제공합니다.
- 분산 환경(Vision PC ↔ Isaac Sim PC) 간 ROS2 통신 상태를 함께 모니터링합니다.

### 4. (확장) 로봇팔 분류 연동
- `cobot3`는 로봇팔을 직접 움직이지 않고, **로봇팔 분류 동작을 위한 비전 판단 결과(`/qr_code`)를 제공**합니다.
- M0609 로봇팔 제어 노드가 `/qr_code`를 구독해 ZONE별 목표 좌표로 이동, VGC10 흡착 그리퍼로 박스를 분류하는 구조로 확장 가능합니다. (자세한 내용은 "🦾 로봇팔 연동 방향" 섹션 참고)

---

## 🏗️ 시스템 설계 (System Architecture)

### 전체 구조
시스템은 **두 대의 PC**에 분산되어 동작합니다.

| PC | Hostname | IP | 역할 |
|---|---|---|---|
| Vision PC | `vision` | 10.0.0.1 | YOLO11 추론, QR 디코딩, PyQt5 GUI, ROS2 허브 노드 |
| Isaac Sim PC | `IsaacSim05` | 10.0.0.2 | Isaac Sim 시뮬레이션 (컨베이어, 카메라, OmniGraph 제어) |

두 PC는 **유선 기가비트 LAN**으로 직접 연결되어 있으며, 정적 IP(netplan) + FastDDS XML 프로필로 유선 인터페이스만 사용하도록 제한되어 있습니다 (설정 방법은 "📦 의존성 → 0. PC 간 LAN 통신 설정" 참고).

### 전체 데이터 흐름

```text
[Isaac Sim / Camera]
        │
        │  /rgb
        ▼
[image_transport]
        │
        │  /rgb/compressed
        ▼
[parcel_hub_node]  ← /state/detector, /state/qr_decoder
        │                  ↑ 재명령(워치독)
        │  /hub/rgb/compressed     /cmd/detection_enable
        ▼                          /cmd/conf_threshold
[parcel_detector_node]
        │
        ├─ /parcel_detections
        ├─ /parcel_detections/annotated
        ├─ /parcel_with_qr
        └─ /parcel_no_label
        │
        ▼
[qr_decoder_node]
        │
        ├─ /qr_code   (ZONE_A ~ ZONE_E)
        └─ /qr_crop_image
        │
        ▼
[parcel_control_gui]  ← /hub/state, /hub/alert (경고)
        │
        ▼
[M0609 / VGC10 Robot Arm Control Node]  (별도 파트, /qr_code 구독)
```

### 시스템 역할 분리

| 구분 | 담당 내용 | 관련 파일 |
|---|---|---|
| 비전 인식 | 택배 박스 및 QR 라벨 감지 | `parcel_detector_node.py` |
| QR 판독 | QR crop 및 ZONE 값 디코딩 | `qr_decoder_node.py` |
| 허브 | 영상 중계, 상태 감시, 명령 중계 | `parcel_hub_node.py` |
| GUI | 영상, QR 결과, 노드 상태 확인 | `parcel_control_gui.py` |
| 이상 감지 (미사용) | PatchCore 기반 박스 훼손 판별 | `patchcore_anomaly_node.py` |
| 로봇팔 동작 (별도 파트) | M0609 + VGC10 흡착 이송 및 분류 | 별도 Isaac Sim py / USD 파일 |

### Isaac Sim 컨베이어 제어
- OmniGraph 변수 `Velocity`를 `graph.find_variable("Velocity")` → `variable.set(...)`으로 제어합니다.
- 방향은 `Sorter/ActionGraph` 내 `ConstantFloat` 노드(`Direction`, 기본값 90.0°)로 제어합니다.

> (참고: ArUco 마커는 개념 검토 단계에서만 고려되었고, 실제 구현/테스트에는 사용되지 않았습니다.)

### 플로우 차트 (Logic Flow)
```
[Isaac Sim 컨베이어 시작]
        │
        ▼
[카메라 입력 / parcel_detector_node]
        │
        ▼
   YOLO11 탐지 ──► 택배 + QR 영역 검출?
        │ No                  │ Yes
        ▼                     ▼
  계속 컨베이어 진행      bbox 크롭 → qr_decoder_node
                              │
                       pyzbar(우선) / OpenCV(fallback) QR 디코딩
                              │
                  ┌───────────┴────────────┐
            ZONE_A ~ ZONE_E             디코딩 실패
                  │                         │
        parcel_hub_node가 상태 갱신     재시도 / NO_QR
          (/hub/state, /qr_code)
                  │
        컨베이어 커터(게이트) 동작 → 분류 완료
```
*(최종 제출 시 위 플로우를 다이어그램 이미지로 별도 첨부 권장)*

---

## 📂 폴더 구조

```text
cobot3/
├── cobot3/
│   ├── __init__.py
│   ├── talker.py                  # ROS2 예제 - PC간 통신 테스트용
│   ├── listener.py                # ROS2 예제 - PC간 통신 테스트용
│   ├── parcel_detector_node.py
│   ├── qr_decoder_node.py
│   ├── patchcore_anomaly_node.py  # (이상 탐지, 실험용)
│   ├── parcel_hub_node.py
│   └── parcel_control_gui.py
├── launch/
│   └── parcel_detector.launch.py
├── models/
│   ├── parcel_qr_det.pt           # 기본 실행 스크립트에서 사용하는 모델
│   ├── parcel_box_baseline.pt
│   ├── parcel_box_conveyor_det.pt
│   ├── parcel_box_isaac_det.pt
│   ├── patchcore_memory_bank.pt   # PatchCore
│   └── patchcore_threshold.pt     # PatchCore
├── scripts/
│   └── start_vision.sh
├── resource/
│   └── cobot3
├── test/
│   ├── test_copyright.py
│   ├── test_flake8.py
│   └── test_pep257.py
├── package.xml
├── setup.py
├── setup.cfg
└── LICENSE
```


---

## 🔍 주요 노드 설명

### 1. `parcel_hub_node` — 중앙 허브

영상 재배포 및 각 노드 상태 감시를 담당하는 중앙 허브 노드입니다.

- `/rgb/compressed` 영상 수신 → `/hub/rgb/compressed`로 재발행
- detector, QR decoder, simulation 상태 감시 (타임아웃: detector 5초 / qr_decoder 8초, 최대 3회 재명령)
- `/hub/state`, `/hub/alert`를 통해 GUI에 상태 정보 제공
- `/cmd/detection_enable`, `/cmd/conf_threshold`, `/cmd/qr_enable`, `/cmd/relay_enable` 명령 처리(중계)

| 방향 | 토픽 | 타입 | 설명 |
|---|---|---|---|
| Subscribe | `/rgb/compressed` | `sensor_msgs/CompressedImage` | 입력 영상 |
| Publish | `/hub/rgb/compressed` | `sensor_msgs/CompressedImage` | 허브 재배포 영상 |
| Subscribe | `/state/detector` | `std_msgs/String` | detector 상태 |
| Subscribe | `/state/qr_decoder` | `std_msgs/String` | QR decoder 상태 |
| Publish | `/hub/state` | `std_msgs/String` | 통합 상태 |
| Publish | `/hub/alert` | `std_msgs/String` | 경고 메시지 |

### 2. `parcel_detector_node` — YOLO 감지

YOLO11 모델로 택배 박스와 QR 라벨을 감지하는 노드입니다.

- 입력 영상에서 `package`, `qr_label` 감지 (`vision_msgs/Detection2DArray`로 발행)
- 박스 ↔ QR 라벨 매칭 → `/parcel_with_qr` 발행 / QR 라벨 없는 박스 → `/parcel_no_label`
- 감지 결과 시각화 이미지(annotated) 발행
- `/cmd/detection_enable`, `/cmd/conf_threshold`로 감지 on/off 및 confidence threshold 제어

| 방향 | 토픽 | 타입 | 설명 |
|---|---|---|---|
| Subscribe | `/hub/rgb/compressed` | `sensor_msgs/CompressedImage` | 허브에서 받은 영상 |
| Publish | `/parcel_detections` | `vision_msgs/Detection2DArray` | 전체 감지 결과 |
| Publish | `/parcel_detections/annotated` | `sensor_msgs/Image` | 시각화 이미지 |
| Publish | `/parcel_with_qr` | `vision_msgs/Detection2DArray` | package + qr_label 매칭 결과 |
| Publish | `/parcel_no_label` | `std_msgs/String` | 송장 또는 QR 미부착 박스 (`NO_LABEL`) |
| Publish | `/state/detector` | `std_msgs/String` | detector 상태 |
| Subscribe | `/cmd/detection_enable` | `std_msgs/Bool` | 감지 on/off |
| Subscribe | `/cmd/conf_threshold` | `std_msgs/Float32` | YOLO confidence threshold 변경 |

### 3. `qr_decoder_node` — QR 판독

YOLO가 감지한 QR 라벨 영역을 crop한 뒤 QR 값을 디코딩하는 노드입니다.

- `/parcel_with_qr`에서 package + qr_label bbox 수신, `/hub/rgb/compressed`에서 최신 프레임 캐시
- QR 영역 crop → `pyzbar` 우선 디코딩, 실패 시 OpenCV `QRCodeDetector`로 fallback
- 디코딩 성공 시 `/qr_code` 발행 (`NO_QR`은 발행하지 않음), crop 이미지는 `/qr_crop_image`로 발행

| 방향 | 토픽 | 타입 | 설명 |
|---|---|---|---|
| Subscribe | `/hub/rgb/compressed` | `sensor_msgs/CompressedImage` | QR crop용 영상 |
| Subscribe | `/parcel_with_qr` | `vision_msgs/Detection2DArray` | package + qr_label bbox |
| Publish | `/qr_code` | `std_msgs/String` | 디코딩된 ZONE 값 |
| Publish | `/qr_crop_image` | `sensor_msgs/Image` | QR crop 확인용 이미지 |
| Publish | `/state/qr_decoder` | `std_msgs/String` | QR decoder 상태 |
| Subscribe | `/cmd/qr_enable` | `std_msgs/Bool` | QR decoder on/off 명령 |

### 4. `parcel_control_gui` — 중앙 제어 GUI

PyQt5 기반 GUI 노드입니다.

- 컨베이어 영상 / YOLO 감지 결과 / QR crop 이미지 / QR 판독 결과 표시
- 허브, detector, QR decoder 상태 확인
- emergency stop, reset, threshold 조절 등 제어 UI 제공

| 토픽 | 설명 |
|---|---|
| `/rgb/compressed` | 컨베이어 입력 영상 |
| `/parcel_detections/annotated` | YOLO 시각화 결과 |
| `/parcel_with_qr` | QR이 포함된 택배 감지 결과 |
| `/parcel_no_label` | QR 또는 송장 미부착 박스 |
| `/qr_crop_image` | QR crop 이미지 |
| `/qr_code` | 최종 ZONE 결과 |
| `/hub/state` | 허브 통합 상태 |
| `/hub/alert` | 시스템 경고 |

> `parcel_control_gui.py`는 `cv2`/`numpy`, `rclpy`/ROS2 메시지 패키지가 없어도 import 에러 없이 앱이 실행되도록 try/except로 방어되어 있습니다(`CV2_AVAILABLE`, `ROS2_AVAILABLE` 플래그). 다만 이 경우 카메라 피드·ROS2 통신 기능은 동작하지 않으므로, 실제 기능 사용을 위해서는 아래 의존성이 모두 필요합니다. 캡처 저장 경로(`~/parcel_captures/`, `qr_crops/`, `parcels/`)는 첫 실행 시 자동 생성됩니다.

### 5. `patchcore_anomaly_node` — 이상 감지 (현재 미사용)

PatchCore 기반 박스 이상 감지 노드입니다. RGB 이미지와 YOLO bbox로 박스 영역을 crop하고, PatchCore memory bank/threshold로 정상·훼손 여부를 판단해 `/parcel_anomaly`로 발행합니다.

**테스트/실험용으로 작성되었으며 기본 `start_vision.sh` 파이프라인에는 포함되어 있지 않습니다.** 최종 제출 시 동작 대상에서 제외합니다.

---

## 💻 개발 환경 (Environment)

- **OS:** Ubuntu 22.04 LTS
- **Middleware:** ROS2 Humble Hawksbill
- **Simulator:** NVIDIA Isaac Sim **5.1.0**
- **Language:** Python 3.10
- **Workspace:** `cobot3_ws` / Package: `cobot3`
- **Domain ID:** `ROS_DOMAIN_ID=103`
- **Display:** X11 (Wayland 비활성화, `WaylandEnable=false`)
- **주요 외부 라이브러리:** OpenCV, PyQt5, Ultralytics YOLO, PyTorch, pyzbar

---

## ⚙️ 사용 장비 (Hardware Setup)

| PC | Hostname | IP | 역할 | CPU | GPU | RAM |
|---|---|---|---|---|---|---|
| Vision PC | `vision` | 10.0.0.1 | Vision/GUI | *(입력 필요)* | *(입력 필요)* | *(입력 필요)* |
| Isaac Sim PC | `IsaacSim05` | 10.0.0.2 | Simulation | *(입력 필요)* | RTX 5080 (Blackwell) | *(입력 필요)* |

**네트워크:** 유선 기가비트 LAN 직결, 정적 IP + FastDDS XML 프로필(유선 인터페이스 제한)

> ⚠️ 위 표의 PC 사양(CPU/GPU/RAM)은 placeholder입니다. 정확한 사양으로 교체해 주세요.

---

## 📦 의존성 (Installation / requirements.txt)

### 0. PC 간 LAN 통신 설정 (사전 준비)

두 PC(`vision`, `IsaacSim05`)는 **유선 기가비트 LAN 직결**로 통신합니다. 다른 의존성 설치 전에 먼저 네트워크를 구성해야 합니다. 초기에는 WiFi로 연결했으나 지연/패킷 손실 문제로 유선으로 전환했습니다.

| 항목 | 내용 |
|---|---|
| 연결 방식 | 유선 기가비트 LAN, PC 간 직결 (스위치 미경유) |
| Vision PC (`vision`) | 10.0.0.1 |
| Isaac Sim PC (`IsaacSim05`) | 10.0.0.2 |
| 유선 인터페이스명 | `enp131s0` (양쪽 PC 동일) |
| IP 할당 방식 | 정적 IP (netplan) |
| DDS 미들웨어 | FastDDS |
| ROS_DOMAIN_ID | `103` |

**a) netplan 고정 IP 설정**

각 PC에서 `/etc/netplan/99-wired-static.yaml` 파일을 생성합니다.

Vision PC (`vision`):
```bash
sudo nano /etc/netplan/99-wired-static.yaml
```
```yaml
network:
  version: 2
  ethernets:
    enp131s0:
      addresses:
        - 10.0.0.1/24
```

Isaac Sim PC (`IsaacSim05`):
```bash
sudo nano /etc/netplan/99-wired-static.yaml
```
```yaml
network:
  version: 2
  ethernets:
    enp131s0:
      addresses:
        - 10.0.0.2/24
```

양쪽 PC에서 권한 설정 후 적용:
```bash
sudo chmod 600 /etc/netplan/99-wired-static.yaml
sudo chmod 600 /etc/netplan/*.yaml
sudo netplan apply
```

> 💾 PC별 yaml 파일은 `99-wired-static-vision.yaml` / `99-wired-static-isaacsim.yaml`로 따로 보관해두고, 각 PC에서 아래처럼 복사해서 적용하는 방식을 사용했습니다.
> ```bash
> # Vision PC
> sudo cp 99-wired-static-vision.yaml /etc/netplan/99-wired-static.yaml
> sudo chmod 600 /etc/netplan/99-wired-static.yaml
> sudo netplan apply
>
> # Isaac Sim PC
> sudo cp 99-wired-static-isaacsim.yaml /etc/netplan/99-wired-static.yaml
> sudo chmod 600 /etc/netplan/99-wired-static.yaml
> sudo netplan apply
> ```
> 설정 후에는 재부팅해도 랜선만 연결하면 자동으로 고정 IP가 잡힙니다.

**b) 연결 확인 (ping)**

Vision PC에서: `ping 10.0.0.2`
Isaac Sim PC에서: `ping 10.0.0.1`
양쪽 모두 응답이 오면 성공입니다.

**c) FastDDS 유선 인터페이스 전용 설정**

각 PC에서 자신의 IP만 화이트리스트에 넣은 FastDDS XML 프로필을 생성합니다. (서로 다른 IP를 사용하므로 PC별로 파일 내용이 다릅니다)

Isaac Sim PC (`IsaacSim05`, IP `10.0.0.2`):
```bash
mkdir -p ~/.ros
cat > ~/.ros/fastdds_wired.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>udp_wired</transport_id>
      <type>UDPv4</type>
      <interfaceWhiteList>
        <address>10.0.0.2</address>
      </interfaceWhiteList>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name="default_profile" is_default_profile="true">
    <rtps>
      <userTransports>
        <transport_id>udp_wired</transport_id>
      </userTransports>
      <useBuiltinTransports>false</useBuiltinTransports>
    </rtps>
  </participant>
</profiles>
EOF
```

Vision PC (`vision`, IP `10.0.0.1`):
```bash
mkdir -p ~/.ros
cat > ~/.ros/fastdds_wired.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>udp_wired</transport_id>
      <type>UDPv4</type>
      <interfaceWhiteList>
        <address>10.0.0.1</address>
      </interfaceWhiteList>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name="default_profile" is_default_profile="true">
    <rtps>
      <userTransports>
        <transport_id>udp_wired</transport_id>
      </userTransports>
      <useBuiltinTransports>false</useBuiltinTransports>
    </rtps>
  </participant>
</profiles>
EOF
```

> ⚠️ `interfaceWhiteList`의 `<address>`는 **자기 자신의 IP**를 넣습니다 (Vision PC → `10.0.0.1`, Isaac Sim PC → `10.0.0.2`).

**d) 환경변수 설정 (양쪽 PC 동일하게)**
```bash
export ROS_DOMAIN_ID=103
export FASTRTPS_DEFAULT_PROFILES_FILE=~/.ros/fastdds_wired.xml
```
실행 시마다 적용되도록 `~/.bashrc`에 추가하거나, 비전 파이프라인 실행 스크립트(`start_vision.sh`) 안에서 source 합니다. `start_vision.sh`는 내부적으로 `export ROS_DOMAIN_ID=103`을 자동 설정합니다.

> ⚠️ **주의:** `FASTRTPS_DEFAULT_PROFILES_FILE`이 `.bashrc`에 항상 설정되어 있으면 로컬 `ros2 bag` 재생 시 충돌이 발생할 수 있습니다. bag 재생 전에는 해당 줄을 임시로 주석 처리하세요.

**e) ROS2 토픽 통신 확인**
```bash
ros2 topic list
ros2 topic hz /front_stereo_camera/left/image_raw   # 예시 토픽
```

**f) 알려진 이슈**
- 비정상 종료(kill -9 등) 후 노드 간 통신 불가 시 → FastDDS 공유메모리 잔여 파일 정리 필요: `sudo rm -rf /dev/shm/fastrtps_*`
- 두 PC의 `ROS_DOMAIN_ID`가 다르면 토픽이 전혀 보이지 않으므로 반드시 동일하게(`103`) 설정
- 두 PC가 서로 다른 IP를 `interfaceWhiteList`에 넣으면(자기 IP가 아닌 상대 IP를 넣는 등) 통신이 되지 않으므로 주의

### 1. ROS2 패키지

```bash
sudo apt update
sudo apt install ros-humble-desktop \
  ros-humble-cv-bridge \
  ros-humble-vision-msgs \
  ros-humble-image-transport \
  ros-humble-compressed-image-transport
sudo apt install python3-opencv python3-numpy python3-pyqt5 python3-pyzbar -y
```
> ⚠️ **`ros-humble-vision-msgs`는 필수입니다.** `parcel_detector_node.py`, `qr_decoder_node.py`, `parcel_control_gui.py` 세 파일 모두 `from vision_msgs.msg import Detection2DArray, ...`를 사용합니다. 누락 시 import 단계에서 바로 실행이 실패합니다.
> `image-transport`, `compressed-image-transport`는 Isaac Sim에서 발행하는 원본 `/rgb`를 `/rgb/compressed`로 변환하는 데 필요합니다 (아래 "3. 영상 압축 변환" 참고).

### 2. Python (pip) 패키지

**a) PyTorch (CUDA)**

RTX 5080(Blackwell)을 사용 중이므로, GPU 가속이 정상 동작하는지 먼저 확인이 필요합니다.

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
- `torch.cuda.is_available()`이 `True`가 아니거나 PyTorch가 설치되어 있지 않다면, [PyTorch 공식 사이트](https://pytorch.org/get-started/locally/)에서 사용 중인 CUDA 버전에 맞는 설치 명령어를 확인 후 설치하세요.
- ⚠️ Blackwell(RTX 50 시리즈)은 비교적 최신 GPU 아키텍처라 구버전 PyTorch/CUDA 조합에서는 인식이 안 되거나 호환성 문제가 발생할 수 있습니다 (Isaac Sim에서도 동일 이슈가 있었음). CUDA 12.x 이상 + 최신 PyTorch 권장.
- ⚠️ **확인 필요:** 실제 설치에 사용한 정확한 pip 명령어(예: `pip install torch --index-url https://download.pytorch.org/whl/cu124`)를 기록해 주세요. 최종 제출본에는 정확한 명령어를 명시해야 재현 가능합니다.

**b) 나머지 Python 패키지**
```bash
pip3 install ultralytics torch torchvision pillow opencv-python-headless numpy PyQt5 pyzbar
```
> ⚠️ `opencv-python`이 아닌 **`opencv-python-headless`** 를 사용해야 PyQt5와의 Qt 플러그인 충돌을 방지할 수 있습니다.
> `ultralytics`는 PyTorch에 의존하므로, 위 a) 단계에서 GPU용 PyTorch를 먼저 설치한 뒤 `ultralytics`를 설치하는 순서를 권장합니다 (순서가 바뀌면 PyTorch가 CPU 버전으로 재설치될 수 있습니다).

**c) pyzbar 시스템 의존성 (libzbar0)**
```bash
sudo apt install libzbar0 -y
```
> `pyzbar`는 내부적으로 시스템 라이브러리 `libzbar0`를 필요로 합니다. 누락 시 import 시점(`ImportError: Unable to find zbar shared library`)에 오류가 발생합니다.

**d) `package.xml` 의존성 보강 필요**

현재 `package.xml`에는 ROS2 기본 의존성 위주로 작성되어 있고, 실행에 필요한 `ultralytics`, `pyzbar`, `PyQt5`, `torch`, `torchvision`, `Pillow` 등의 외부 Python 패키지는 명시되어 있지 않습니다. `colcon build`만으로는 이 패키지들이 설치되지 않으므로, 위 pip 명령을 **별도로** 실행해야 합니다. 최종 제출 전 `package.xml`/`requirements.txt`에 보강 권장.

### 3. YOLO11 가중치 파일

학습된 YOLO11 가중치(`.pt`)는 `cobot3/models/` 폴더 내에 포함되어 있습니다. 기본 실행 스크립트(`start_vision.sh`)는 **`parcel_qr_det.pt`** 를 사용합니다.

| 파일명 | 용도 |
|---|---|
| `parcel_qr_det.pt` | 택배 박스 + QR 라벨 감지용 **기본 모델** (start_vision.sh 사용) |
| `parcel_box_baseline.pt` | 박스 감지 baseline 모델 |
| `parcel_box_conveyor_det.pt` | 컨베이어 환경 박스 감지 모델 |
| `parcel_box_isaac_det.pt` | Isaac Sim 합성데이터 학습 박스 감지 모델 |
| `patchcore_memory_bank.pt`, `patchcore_threshold.pt` | PatchCore 이상 탐지용 (현재 미사용) |

별도 다운로드 절차 없이 zip 압축 시 함께 포함되므로, 압축 해제 후 바로 추론 가능합니다.

### 4. 영상 압축 변환 (image_transport republish)

Isaac Sim이 발행하는 원본 이미지 토픽(`/rgb`)을 허브가 받을 수 있는 압축 포맷(`/rgb/compressed`)으로 변환해야 합니다. **Vision PC(`vision`)** 에서 `start_vision.sh` 실행 시 자동으로 함께 실행되며, 단독 실행 시에는 다음 명령을 사용합니다.

```bash
source /opt/ros/humble/setup.bash
ros2 run image_transport republish raw \
  --ros-args \
  --remap in:=/rgb \
  --remap out/compressed:=/rgb/compressed
```
- `in:=/rgb`: Isaac Sim ROS2 Bridge가 발행하는 원본 이미지 토픽
- `out/compressed:=/rgb/compressed`: `parcel_hub_node`가 구독하는 입력 토픽 (`input_topic` 파라미터 기본값과 일치)

### 5. Isaac Sim ROS2 Bridge 관련
- `LD_LIBRARY_PATH`는 `isaac_python` alias 안에서만 설정 (전역 `~/.bashrc` 적용 금지 — spdlog 심볼 충돌 방지)

---

## 🔨 빌드 방법

워크스페이스 구조는 다음과 같이 두는 것을 권장합니다.

```text
cobot3_ws/
└── src/
    └── cobot3/
```

빌드는 워크스페이스 루트에서 실행합니다.

```bash
cd ~/cobot3_ws
colcon build --symlink-install
source install/setup.bash
```

ROS2 Humble 환경이 적용되지 않은 새 터미널에서는 먼저 다음을 실행합니다.

```bash
source /opt/ros/humble/setup.bash
source ~/cobot3_ws/install/setup.bash
```

---

## 🚀 실행 순서 (How to Run)

### 1. (Isaac Sim PC: `IsaacSim05`) Isaac Sim 실행

먼저 Isaac Sim에서 카메라 토픽(`/rgb`)이 발행되어야 합니다.

```bash
# Isaac Sim 실행 후 시뮬레이션 씬 로드
# ROS2 Bridge 활성화 확인
isaac_python <시뮬레이션 스크립트>.py
```

### 2. (Vision PC: `vision`) 자동 실행 스크립트 사용 — `start_vision.sh`

`scripts/start_vision.sh`는 tmux를 이용해 비전 파이프라인 전체를 한 번에 실행합니다.

```bash
cd ~/cobot3_ws/src/cobot3
chmod +x scripts/start_vision.sh
./scripts/start_vision.sh
```

워크스페이스 경로는 스크립트 위치 기준으로 동적으로 탐색되므로(`cobot3_ws`라는 이름이 아니어도 동작), 별도 경로 수정이 불필요합니다. 실행 전 모델 파일(`models/parcel_qr_det.pt`)과 워크스페이스 빌드 여부(`install/setup.bash`)를 자동으로 검사합니다.

실행되는 구성(tmux 패널, staggered delay)은 다음과 같습니다.

| 순서 | 패널 | 컴포넌트 | 실행 방식 | 지연 |
|---|---|---|---|---|
| 1 | 0 | `image_transport republish` | `ros2 run` | 즉시 |
| 2 | 1 | `parcel_hub_node` | `ros2 run` | 2초 |
| 3 | 2 | `parcel_detector_node` (`parcel_qr_det.pt`) | `ros2 launch` | 4초 |
| 4 | 3 | `qr_decoder_node` | `ros2 run` | 6초 |
| 5 | 4 | `parcel_control_gui` | `ros2 run` | 10초 |

기본 `ROS_DOMAIN_ID`는 스크립트 내부에서 `103`으로 설정되어 있습니다. Isaac Sim 또는 다른 ROS2 노드와 통신하려면 동일한 `ROS_DOMAIN_ID`를 사용해야 합니다.

> tmux 단축키: 패널 전환 `Ctrl+B → 화살표키`, 세션 나가기(백그라운드 유지) `Ctrl+B → D`
> ⚠️ Isaac Sim은 `start_vision.sh`에 포함되지 않으므로, 위 1단계에서 **별도 터미널(Isaac Sim PC)로 먼저 실행**해야 합니다.

### 3. 수동(개별) 실행

각 노드를 직접 실행할 수도 있습니다. 먼저 환경을 source합니다.
```bash
source /opt/ros/humble/setup.bash
source ~/cobot3_ws/install/setup.bash
export ROS_DOMAIN_ID=103
```

```bash
# [1/5] image_transport
ros2 run image_transport republish raw \
  --ros-args --remap in:=/rgb --remap out/compressed:=/rgb/compressed

# [2/5] parcel_hub_node
ros2 run cobot3 parcel_hub_node \
  --ros-args \
  -p input_topic:=/rgb/compressed \
  -p output_topic:=/hub/rgb/compressed \
  -p enable_watchdog:=true

# [3/5] parcel_detector_node (launch)
ros2 launch cobot3 parcel_detector.launch.py \
  model_path:=$(ros2 pkg prefix cobot3)/share/cobot3/models/parcel_qr_det.pt \
  rgb_topic:=/hub/rgb/compressed

# [4/5] qr_decoder_node
ros2 run cobot3 qr_decoder_node \
  --ros-args \
  -p rgb_topic:=/hub/rgb/compressed \
  -p publish_only_on_change:=false

# [5/5] GUI
ros2 run cobot3 parcel_control_gui
```

> ⚠️ `qr_decoder_node` 실행 시 `-p publish_only_on_change:=false` 파라미터를 사용하는데, 코드 리뷰 결과 `qr_decoder_node.py`에 이 파라미터를 `declare_parameter`하는 부분이 확인되지 않았습니다. 코드 버전 불일치 가능성이 있으니, 실행 시 "unknown parameter" 경고가 뜨는지 확인 권장.

### 트러블슈팅 참고
- LAN/DDS 통신 관련 트러블슈팅은 위 "의존성 → 0. PC 간 LAN 통신 설정" 섹션 참고

---

## 🦾 로봇팔 연동 방향 (별도 파트)

`cobot3` 패키지는 카메라 영상에서 QR 값을 읽어 `/qr_code` 토픽으로 ZONE 결과를 발행합니다. 예상 데이터 예시는 다음과 같습니다.

```text
ZONE_A
ZONE_B
ZONE_C
ZONE_D
ZONE_E
```

로봇팔 제어 파트는 이 `/qr_code` 값을 구독한 뒤, 각 ZONE에 대응되는 목표 좌표로 M0609 로봇팔을 이동시키고 VGC10 흡착 그리퍼로 박스를 집어 분류하는 방식으로 연결할 수 있습니다.

```text
cobot3 비전 패키지                       M0609 로봇팔 제어 파트
  - 카메라 영상 수신                       - /qr_code 구독
  - 택배 박스 / QR 라벨 감지        ──►     - ZONE별 목표 위치 선택
  - QR 디코딩                              - VGC10 흡착
  - /qr_code 발행                          - 박스 이동 → 흡착 해제
```

예시 목표 좌표 구조 (개념 코드):
```python
ZONE_TARGETS = {
    "ZONE_A": (0.32,  0.25, 0.10),
    "ZONE_B": (0.32,  0.10, 0.10),
    "ZONE_C": (0.32,  0.00, 0.10),
    "ZONE_D": (0.32, -0.10, 0.10),
    "ZONE_E": (0.32, -0.25, 0.10),
}
```

연동 흐름:
```text
1. cobot3가 QR을 인식한다.
2. qr_decoder_node가 /qr_code로 ZONE 값을 발행한다.
3. 로봇팔 제어 노드가 /qr_code를 구독한다.
4. ZONE 값에 따라 목표 좌표를 선택한다.
5. M0609가 박스 위치로 이동한다.
6. VGC10 흡착 그리퍼가 박스를 흡착한다.
7. 목표 ZONE 위치로 이동한다.
8. 흡착을 해제하고 다음 박스를 대기한다.
```

기존 M0609 / VGC10 / Isaac Sim 파일과 연결할 때 맞춰야 할 항목:

| 항목 | 설명 |
|---|---|
| 카메라 토픽 | Isaac Sim에서 `/rgb` 또는 지정된 카메라 토픽 발행 |
| QR 결과 토픽 | `cobot3`에서 `/qr_code` 발행 |
| 로봇 제어 입력 | 로봇팔 노드에서 `/qr_code` 구독 |
| 목표 좌표 | `ZONE_A` ~ `ZONE_E`별 적재 또는 분류 위치 |
| 그리퍼 동작 | VGC10 또는 Surface Gripper 흡착 on/off |
| USD 환경 | 컨베이어, 박스, 카메라, M0609, VGC10 prim 경로 일치 |

따라서 `cobot3`는 **로봇팔을 직접 움직이는 패키지라기보다는 로봇팔 분류 동작을 위한 비전 판단 결과를 제공하는 패키지**입니다.

---

## ⚠️ 현재 확인된 주의사항

### 1. GUI 제어 토픽 이름 불일치 (확인 필요)

`parcel_control_gui.py` 코드 일부는 다음 토픽을 사용합니다.
```text
/detection_enable
/yolo_conf_threshold
```
하지만 detector와 hub는 다음 토픽을 사용합니다.
```text
/cmd/detection_enable
/cmd/conf_threshold
```
GUI에서 detection on/off나 confidence threshold 조절이 실제 detector에 반영되지 않는다면, GUI의 토픽명을 `/cmd/...` 형식으로 맞춰야 합니다.

권장 수정 방향:
```python
CONF_TOPIC = "/cmd/conf_threshold"
DETECT_EN_TOPIC = "/cmd/detection_enable"
```

### 2. `/parcel_no_label` 타입 확인 필요

`parcel_detector_node.py`는 `/parcel_no_label`을 `std_msgs/String`(`"NO_LABEL"`)으로 발행합니다. 반면 GUI 코드에서는 이 토픽을 이미지 토픽처럼 처리하는 부분이 있어 실행 시 타입 불일치가 발생할 수 있습니다. GUI에서 이 토픽을 사용할 경우 `std_msgs/String` 기준으로 맞추는 것이 안전합니다.

### 3. QR enable 동작 확인 필요

`qr_decoder_node.py`에는 `/cmd/qr_enable` 구독자가 있지만, 실제 QR 처리 콜백에서 `_enabled` 상태값을 기준으로 처리 차단이 완전히 적용되어 있는지 확인이 필요합니다.

### 4. `package.xml` 의존성 누락

위 "📦 의존성 → 2-d) `package.xml` 의존성 보강 필요" 참고.

### 5. YOLO 모델 경로 / 클래스 ID 기본값

`parcel_detector_node.py`의 `model_path` 파라미터 기본값은 `yolo11n.pt`(범용 COCO 모델)이며, `target_class_ids` 기본값도 `[28]`(COCO `suitcase`)로 되어 있어 커스텀 학습 모델(`parcel_qr_det.pt` 등) 사용 전 임시 설정입니다. 실제 실행 시(launch/스크립트에서) `model_path`를 올바른 `.pt`로, 클래스 ID를 `package`/`qr_label`에 맞게 지정하는지 확인이 필요합니다.

---

## 📝 기타 비고
- 게이트 제어는 `/simulation_control` 토픽이 아닌 **컨베이어 커터(물리적 게이트)** 로 수행됩니다.
- FastAPI, `/sort_cmd` 토픽은 현재 미사용입니다.
- ArUco 마커는 구현/테스트되지 않았습니다 (개념 검토만 진행).
- `cobot3/cobot3/listener.py`, `talker.py`는 ROS2 기본 예제 잔재 파일로 보입니다 — 실제 사용 여부 확인 후 미사용이면 제출 전 삭제 권장.

---

## 🗂️ Git 업로드 시 제외 권장 파일

다음 파일들은 용량이 크거나 개인 PC 환경에 종속적이므로 Git에 올리지 않는 것을 권장합니다.

```text
.vscode/browse.vc.db
**/.vscode/browse.vc.db
__pycache__/
*.pyc
build/
install/
log/
parcel_captures/
```

특히 `.vscode/browse.vc.db`는 VSCode가 자동 생성하는 인덱스 캐시 파일이므로 프로젝트 실행에 필요하지 않습니다.

### 예시 `.gitignore`
```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/

# ROS2 build outputs
build/
install/
log/

# VSCode cache
.vscode/browse.vc.db
**/.vscode/browse.vc.db
.vscode/ipch/
**/.vscode/ipch/

# Local captures
parcel_captures/
```

> 제출 가이드라인 기준으로는, `build`/`install`/`log` 폴더를 삭제한 뒤 **`src` 폴더가 포함된 워크스페이스 전체**를 압축해 제출합니다. `cobot3_ws/src/basic/`(개인 실습 코드)도 함께 제외합니다.

---

## 🔭 향후 개선 방향
- GUI 제어 토픽을 `/cmd/...` 형식으로 통일
- `/parcel_no_label` 토픽 타입 정리
- QR enable/disable 명령의 실제 처리 반영
- M0609 로봇팔 제어 노드와 `/qr_code` 연동
- `ZONE_A` ~ `ZONE_E`별 목표 좌표 및 적재 알고리즘 정리
- VGC10 흡착 성공 여부를 상태 토픽으로 발행
- 송장 미부착 또는 QR 인식 실패 박스의 예외 처리 구역 추가
- PatchCore 이상 감지 노드를 기본 파이프라인에 선택적으로 통합
- `package.xml`에 실제 실행 의존성 보강

---

## 핵심 요약

`cobot3`는 택배 분류 시스템에서 비전 인식과 QR 판독을 담당하는 ROS2 패키지입니다. Vision PC(`vision`)와 Isaac Sim PC(`IsaacSim05`)가 유선 LAN으로 연결된 분산 환경에서, 카메라 영상으로부터 택배 박스와 QR 라벨을 감지하고 QR 값을 `/qr_code`로 발행합니다. 이 결과를 M0609 로봇팔 제어 노드가 구독하면, VGC10 흡착 그리퍼를 이용해 박스를 ZONE별로 분류하는 전체 자동화 시스템으로 확장할 수 있습니다.

---

*본 문서는 제출 가이드라인 기준 임시 통합 작성본입니다. 최종 제출 전 PC 사양, PyTorch/CUDA 설치 명령어, 토픽 불일치 항목, package.xml 의존성을 검증 및 보완해 주세요.*

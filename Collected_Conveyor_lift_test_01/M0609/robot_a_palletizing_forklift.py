from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# 122_: 새 Conveyor_lift.usd 환경 적용 버전.
#       Omni/Hydra material compile warning이 콘솔을 도배할 수 있어,
#       Traceback/Error가 아닌 일반 Warning은 최대한 줄인다.
SUPPRESS_NON_CRITICAL_KIT_WARNINGS_122 = True
if SUPPRESS_NON_CRITICAL_KIT_WARNINGS_122:
    try:
        import carb
        _settings = carb.settings.get_settings()
        for _key in (
            "/log/level",
            "/log/outputStreamLevel",
            "/log/fileLogLevel",
            "/log/consoleLogLevel",
        ):
            try:
                _settings.set(_key, "error")
            except Exception:
                pass
    except Exception as _log_filter_e:
        print(f"[122_LOG_FILTER][WARN] log filter setup skipped: {_log_filter_e}")

from isaacsim.core.utils.extensions import enable_extension

# 108_: 106번 성공 상태를 기반으로 link_2 Orient Z guard를 추가하고, lift -> swing -> release 팔레타이징 테스트를 수행한다.
#      ready zone A에서 OriBoxA가 정지하면 suction point 기준으로 흡착 판정 후
#      link_6 <-> OriBox root FixedJoint를 만들고, 박스 좌표를 코드로 직접 움직이지 않고 joint_2 제한 + joint_3 보조 lift + joint_1 swing을 테스트한다.
#      link_2 Orient Z가 초기 -90도보다 더 작아지지 않도록 joint_2 목표값을 계산/제한한다.
#      박스 좌표/정지/흡착/이동 기준은 child가 아니라 OriBox root이다.
#      1번 로봇=/World/m0609_A 를 active로 사용하고, 2번 로봇=/World/m0609_B 는 아직 제어하지 않는다.
# USD 안의 rsd455 카메라 prim은 더 이상 비활성화하지 않는다.
# ROS2 토픽 publish까지 필요하면 True로 바꾸고 ROS2 환경을 source 한 뒤 실행한다.
ENABLE_ROS2_BRIDGE = False
if ENABLE_ROS2_BRIDGE:
    enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# 134_: 130번 기반. release 후 task_done에 막히지 않고 로봇이 home/초기 자세로 복귀하도록 수정한 버전.
# 143_: /World/APalt 가상 지게차 이동에 X축 이동 옵션을 추가. lift -> X -> Y 순서로 천천히 이동.
# 147_: USD root의 /Environment를 /World reference와 별도로 현재 stage root /Environment에 reference해서 보이게 함.
# 154_: 147 기준 유지. 흡착 전 정중앙/pre-align 과정은 건드리지 않고, 흡착 이후 lift/swing/lower/release/home 구간만 약 20~30% 단축.
# 159_: slot별 joint_1 회전 방향 분리 + APalt_slot 방향(yaw) 진단.
# 171_: 170 안정 흐름 유지 + release 직전 APalt_slot 기준 yaw만 RMPFlow pose-align으로 짧게 보정.

from pathlib import Path
import sys
import time
import math

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.robot.manipulators.manipulators import SingleManipulator
try:
    from isaacsim.core.utils.types import ArticulationAction
except Exception:
    from omni.isaac.core.utils.types import ArticulationAction

_THIS_DIR = Path(__file__).resolve().parent

# rmpflow 인프라 폴더 경로 등록 (인프라 파일 내부 import가 그대로 동작)
RMPFLOW_DIR = str(_THIS_DIR / "rmpflow")
if RMPFLOW_DIR not in sys.path:
    sys.path.insert(0, RMPFLOW_DIR)
 
from m0609_pick_place_controller import PickPlaceController
from m0609_rmpflow_controller import RMPFlowController

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. Task 파라미터                                             ║
# ╚══════════════════════════════════════════════════════════════╝
# 원본 M0609 USD를 로드한다.
# RG2 삭제와 VGC10 부착은 실행 중 Stage에서만 수행되므로 원본 USD 파일은 수정되지 않는다.
PROJECT_DIR = _THIS_DIR.parent
# Conveyor 프로젝트 구조 기준:
# Conveyor/
# ├── M0609/      ← 이 py 파일 위치
# ├── conveyor/   ← 환경 USD 모음 폴더
# ├── oriA/OriBoxA.usda
# └── oriB/OriBoxB.usda
#
# 절대경로를 쓰지 않는다. Conveyor 폴더 전체를 다른 PC/다른 위치로 옮겨도
# M0609/ 과 conveyor/ 의 상대 구조만 유지하면 그대로 동작한다.
def _resolve_first_existing_path(candidates, label="file"):
    for p in candidates:
        pp = Path(p)
        if pp.exists():
            return str(pp)
    return str(Path(candidates[0]))

CONVEYOR_USD_CANDIDATES = [
    # 122_: 사용자가 새로 수정한 작업환경 USD를 우선 사용한다.
    # py 위치가 .../Collected_Conveyor_lift/M0609 이면 PROJECT_DIR은 .../Collected_Conveyor_lift 이다.
    # 따라서 새 USD는 보통 ~/Videos/Collected_Conveyor_lift/Conveyor_lift.usd 에 두면 된다.
    PROJECT_DIR / "Conveyor_lift.usd",
    _THIS_DIR / "Conveyor_lift.usd",

    # fallback: 새 파일이 없으면 기존 표준 USD 사용.
    PROJECT_DIR / "Conveyor_lift.usd",
    _THIS_DIR / "Conveyor_lift.usd",
]
CONVEYOR_USD_PATH = _resolve_first_existing_path(CONVEYOR_USD_CANDIDATES, "Conveyor USD")
USD_PATH        = CONVEYOR_USD_PATH
# 40_ 핵심 변경:
# 70_ 핵심 변경:
# 새 Conveyor_lift.usd에서는 1번 로봇이 /World/m0609_A 이다.
# 지금은 1번 로봇만 동작 테스트하고, /World/m0609_B 는 idle 로봇으로 둔다.
ACTIVE_ROBOT_ROOT_PATH = "/World/m0609_A"
ROBOT_PRIM_PATH = ACTIVE_ROBOT_ROOT_PATH + "/m0609"
EE_LINK_NAME    = "link_6"
print(f"[122_ENV] Conveyor USD selected: {USD_PATH}")
print(f"[122_ENV] active robot root: {ACTIVE_ROBOT_ROOT_PATH}  (1번 로봇=A)")
print("[122_ENV] idle robot expected: /World/m0609_B  (2번 로봇, 아직 제어 없음)")
print("[122_ENV] rsd455 camera restore enabled: /World/rsd455 will NOT be disabled")
print("[155_SLOT_MARKER] APalt slot marker palletizing enabled: /World/APalt_slot_01, /World/APalt_slot_02")
print("[155_SLOT_MARKER] flow: center suction -> safe lift above conveyor boxes -> joint_1 coarse swing -> marker final move/lower -> release")
print("[185_FOUR_BOX] 4개 연속 적재 모드: 1~4번째 모두 lower/settle 중 yaw 정렬을 같이 수행하고 2번째에서 멈추지 않음")

# 기존 RG2 집게 삭제 대상. 7_ 버전은 reference prim까지 강제 제거/비활성화한다.
OLD_GRIPPER_PRIM_PATH = ROBOT_PRIM_PATH + "/onrobot_rg2ft"

# idle 로봇 설정: /World/m0609_* 중 active가 아닌 로봇에 VGC10 visual을 붙이고 제어하지 않는다.
ATTACH_VGC10_TO_IDLE_M0609_ROBOTS = False  # 122_: 새 USD 환경 보존. 2번/idle 로봇에 VGC10 visual을 새로 붙이지 않음
IDLE_M0609_ROOT_PREFIX = "m0609_"
IDLE_VGC10_ROOT_PREFIX = "/World/vgc10_visual_idle"

# 실제 VGC10 CAD를 Blender/Isaac 등에서 변환한 USDA 경로.
# .usda는 ASCII 형태의 USD라서 .usd로 변환하지 않아도 AddReference로 바로 붙일 수 있다.
# 핵심 수정: VGC10을 link_6/tool0 하위에 직접 넣지 않고 /World 아래 visual-follow 모델로 둔다.
# 이렇게 해야 CAD mesh/collider가 로봇 articulation에 섞이면서 바닥을 밀거나 집는 현상을 피할 수 있다.
VGC10_USD_PATH  = str(_THIS_DIR / "assets/gripper_vgc10_v1.usda")
VGC10_FALLBACK_PATHS = [
    str(_THIS_DIR / "assets/gripper.usda"),
    str(_THIS_DIR / "assets/gripper_vgc10.usd"),
    str(_THIS_DIR / "assets/gripper_vgc10_v1.usda"),
]
VGC10_PRIM_PATH = "/World/vgc10_visual_follow"
# scale이 확실히 적용되도록, 움직이는 root와 스케일 적용 mount를 분리한다.
# root: tool0 월드 위치/자세를 따라감
# mount: gripper 모델에 local translate/rotate/scale 적용
# model: 실제 gripper.usda reference
VGC10_MOUNT_PATH = "/World/vgc10_visual_follow/vgc10_scaled_mount"
VGC10_MODEL_PATH = "/World/vgc10_visual_follow/vgc10_scaled_mount/gripper_model"
VGC10_FOLLOW_TARGET_PATH = ROBOT_PRIM_PATH + "/" + EE_LINK_NAME + "/tool0"

# 106_: 카메라 복구 설정.
# 105번까지는 오류 회피용으로 /World/rsd455 자체를 비활성화했지만,
# 이제 rsd455가 필요하므로 카메라 prim은 active 상태로 유지한다.
RSD455_ROOT_PATH = "/World/rsd455"
KEEP_RSD455_CAMERA_ACTIVE = True
DISABLE_LEGACY_ROBOT_CAMERA_GRAPHS = False  # 122_: 새 USD에 저장한 camera_graph/환경을 비활성화하지 않음
DISABLE_WORLD_RSD455_PRIM = False

# 109_: 사용자가 Stage에서 선택한 Cube가 py 실행 후 화면에서 사라지는 문제 방지.
# 사진 기준으로 Prim Path가 /Cube로 보이므로 /Cube와 /World/Cube를 모두 강제로 visible/active 처리한다.
FORCE_SHOW_CUBE_PRIMS_109 = True
CUBE_VISIBLE_CANDIDATE_PATHS_109 = ("/Cube", "/World/Cube", "/World/APalt")  # 122_: APalt도 실행 중 active/visible 보장

# 146_: 새 USD에서 Prim Path가 Environment인 환경 오브젝트들이 145 실행 후 안 보이는 문제 방지.
# /World에 USD를 reference하면 원본 /Environment가 /World/Environment로 들어올 수 있으므로 둘 다 처리한다.
FORCE_SHOW_ENVIRONMENT_PRIMS_146 = True
ENVIRONMENT_VISIBLE_CANDIDATE_PATHS_146 = (
    "/Environment",
    "/World/Environment",
    "/World/environment",
    "/World/ENVIRONMENT",
)
ENVIRONMENT_VISIBLE_NAME_KEYWORDS_146 = ("Environment", "environment", "ENVIRONMENT")
ENVIRONMENT_FORCE_PURPOSE_DEFAULT_146 = True


# VGC10 local pose 보정값. 모델 축/크기에 따라 여기만 조정하면 된다.
# gripper.usda는 네가 확인한 기준으로 X=-90도가 제대로 보이는 방향이다.
VGC10_LOCAL_TRANSLATE = np.array([0.0, 0.0, 0.0], dtype=float)
VGC10_LOCAL_ROTATE_XYZ = np.array([-90.0, 0.0, 0.0], dtype=float)
VGC10_LOCAL_SCALE = np.array([0.001, 0.001, 0.001], dtype=float)

# VGC10 실제 흡착 기준점 보정값.
# 이제 scripted_suction_body / scripted_suction_direction_marker visual은 만들지 않는다.
# 이 값은 /World/vgc10_visual_follow 기준 local offset이며, tool0/VGC10을 따라 같이 움직인다.
# 안 붙으면 VSCode에서 이 값만 조정한다.
VGC10_SUCTION_POINT_PATH = "/World/vgc10_visual_follow/vgc10_suction_point"
# GUI에서 맞춘 suction point 위치. 이 값은 VGC10 기준 local offset이다.
VGC10_SUCTION_LOCAL_OFFSET = np.array([0.0, 0.0, 0.12476], dtype=float)

# 51_ 핵심: 단일 흡착점이 아니라 VGC10 기준 3x3 흡착점 그리드를 코드에서 만든다.
# 전체 폭은 8cm x 8cm. 박스 윗면을 너무 넓게 벗어나지 않으면서도 한 점 흡착보다 안정적으로 판정한다.
SUCTION_GRID_ENABLED = False  # 59_: 9점 판정 중지. 중심 흡착점 1개 + 넓은 흡착 반경으로 복구
SUCTION_GRID_ROOT_PATH = "/World/vgc10_visual_follow/vgc10_suction_grid"
SUCTION_GRID_SPACING_XY = 0.060  # 59_: 디버그용만. 실제 판정은 중심 흡착점 1개로 처리
SUCTION_GRID_MARKERS_VISIBLE = True
SUCTION_GRID_MARKER_RADIUS = 0.007
SUCTION_GRID_ATTACH_MIN_POINTS = 1  # 59_: 중심 흡착점 1개 판정
SUCTION_GRID_ATTACH_REQUIRE_CENTER = True
SUCTION_GRID_LOG_INTERVAL = 1  # 52_: 흡착점이 제대로 동작하는지 매 step 가까이 로그 확인
SUCTION_GRID_LOG_EVERY_STEP = True  # 52_: p00~p22 hit 여부를 확실히 확인
SUCTION_GRID_VERBOSE_POINTS = True

# 59_: 9점 흡착 판정을 중지하고, 중심 흡착점 1개를 넓은 원형 흡착 패드처럼 취급한다.
# - 실제 흡착 위치는 p_center 1개
# - XY는 박스 윗면 중심 근처인지 확인
# - Z는 실제 윗면에 거의 닿았는지 확인
CENTER_SUCTION_SINGLE_POINT_MODE = True
CENTER_SUCTION_EFFECTIVE_RADIUS_XY = 0.085  # m. 실제 패드가 넓다고 가정하는 반경
CENTER_SUCTION_CENTER_TOL_XY = 0.055        # m. 중심점이 box_top 중심에서 허용되는 오차
CENTER_SUCTION_LOG_INTERVAL = 1


# 51_ 핵심: 흡착 성공 시 박스 물리를 끄지 않고 link_6와 박스 rigid body를 FixedJoint로 연결한다.
# 실패하면 joint path/body 경로 로그가 뜬다.
PHYSICS_FIXED_JOINT_ATTACH_ENABLED = True
PHYSICS_ATTACH_JOINT_PATH = "/World/vgc10_physics_attach_fixed_joint"
PHYSICS_ATTACH_BODY0_LINK_NAME = "link_6"
PHYSICS_ATTACH_BODY1_USE_BOX_PRIM = True
PHYSICS_RELEASE_ZERO_VELOCITY = True

# 흡착 위치 확인이 필요할 때만 True로 바꾼다. 기본값 False라 화면에 빨간/흰 마커가 안 보인다.
DEBUG_SHOW_SUCTION_POINT = False
VGC10_SUCTION_DEBUG_MARKER_PATH = "/World/vgc10_suction_debug_marker"

# 29_ 정리: 목표 위치 표시용 goal_marker는 더 이상 만들지 않는다.
# USD에 이미 들어있어도 실행 시 숨김/비활성화한다.
SHOW_GOAL_MARKER = True  # 122_: 새 USD에서 직접 배치한 goal_marker를 삭제/숨김 처리하지 않음
GOAL_MARKER_PATH = "/World/goal_marker"

# VGC10 CAD/USD는 화면에만 보여주고 물리 충돌/rigid body는 제거한다.
# 그래야 VGC10 모델이 바닥이나 큐브를 밀어 물리가 꼬이지 않는다.
VGC10_VISUAL_ONLY = True

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING   = 1e4
DRIVE_MAX_FORCE = 1e8

# 기존 finger gripper는 완전히 제거한다. 컨트롤러에는 NullGripper를 넘긴다.
CUBE_STATIC     = 1.2
CUBE_DYNAMIC    = 1.0

# 흡착 시연 로직. 실제 visual 마커는 만들지 않고, VGC10 suction point 좌표만 계산한다.
# 기존 scripted_suction_body / scripted_suction_direction_marker는 생성하지 않는다.
SUCTION_BODY_PATH = "/World/scripted_suction_body"              # 이전 버전 청소용 경로
SUCTION_MARKER_PATH = "/World/scripted_suction_direction_marker" # 이전 버전 청소용 경로
TARGET_CUBE_PATH = "/World/OriBoxA_01"  # 기존 코드 호환용 이름. 72_에서는 실제 대상도 root BOX_PRIM_PATH를 사용한다.
SUCTION_OFFSET_FROM_EE = np.array([0.0, 0.0, -0.12])  # fallback 전용
# 23_ 기준: suction_pos 계산용 안전 하한.
# 주의: 이 값은 로그/판정용 suction_pos 하한이고, VGC10 visual 자체의 실제 위치를 멈추는 값은 아니다.
SUCTION_MIN_Z = 0.070
# 네가 직접 맞춘 값. 이제 이 값은 고정하고, 흡착 타이밍은 should_script_attach()의 ok 조건만 조정한다.
SUCTION_HOLD_OFFSET = np.array([0.0, 0.0, -0.056])
# 21_ 수정: 실제 흡착은 접촉 직전에만 붙는 게 아니라 약간 떨어진 거리에서도 빨아들일 수 있다.
# 그래서 event=1 후반부터 흡착 판정을 허용한다.
# 단, hold_error 기준을 사용해서 너무 멀거나 너무 눌러 들어간 상태는 막는다.
PICK_CLOSE_EVENTS = {1, 2, 3}
RETRY_CLOSE_EVENTS = {4}
FAIL_IF_NOT_ATTACHED_EVENT = 5

# 23_ 핵심: 마지막 place 하강(event=6)에서 큐브를 계속 따라 내리면
# VGC10 visual이 큐브를 뚫고 내려가는 것처럼 보인다.
# 그래서 event=6에서 suction point가 이 높이 이하로 내려오면 바로 목표 위치에 내려놓고 정지한다.
PLACE_RELEASE_EVENT = 5  # 23_: event=5에서 반대편 XY 근처에 도착하면 바로 흡착 해제
PLACE_RELEASE_SUCTION_Z = 0.095
PLACE_APPROACH_MIN_CUBE_Z = None  # CUBE_INIT_POS 정의 후 아래에서 설정
PAUSE_AFTER_RELEASE = False
RETURN_HOME_AFTER_RELEASE = True
HOME_RETURN_STEPS = 160

# ╔══════════════════════════════════════════════════════════════╗
# ║  B. Controller 파라미터                                       ║
# ╚══════════════════════════════════════════════════════════════╝
M0609_URDF_PATH           = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH    = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

CUBE_INIT_POS = np.array([1.92428, -4.20444, -0.41169])  # fallback/screenshot local translate, 실제 판정은 bbox world 사용
CUBE_HALF_Z   = 0.12  # ori box scale 2 기준 대략 half height fallback
PLACE_APPROACH_MIN_CUBE_Z = float(CUBE_INIT_POS[2] + 0.010)
GOAL_POS      = np.array([0.0, -0.45, 0.0])  # 새 코드에서는 박스 기준 상대 offset으로 사용
# 21_ 기준 유지: 0.15는 너무 낮아서 흡착 전에 VGC10이 큐브를 뚫었다.
# 0.17부터 테스트한다. 더 내려가야 하면 0.165, 아직 뚫으면 0.175로 조정한다.
EE_OFFSET     = np.array([0.0, 0.0, 0.17])

EVENTS_DT = [
    0.008,   # 0. 접근 이동
    0.005,   # 1. 하강
    0.35,    # 2. 흡착 ON 단계: dt는 작을수록 느리다. 0.35면 약 3 step 후 다음 event
    0.80,    # 3. 흡착 유지 대기 최소화: 바로 다음 단계로 넘어가게 함
    0.025,   # 4. retry window: 흡착 실패 시 바로 출발하지 않고 근접 상태를 조금 더 유지
    0.01,    # 5. Place 위치로 이동
    0.0025,  # 6. 하강: 23_에서는 너무 낮아지기 전에 조기 release + pause로 침투를 막는다.
    0.2,     # 7. 흡착 OFF 대기 단축
    0.008,   # 8. 상승
    0.08,    # 9. 복귀
]

# ╔══════════════════════════════════════════════════════════════╗
# ║  C. Conveyor / OriBoxA 대상 설정                              ║
# ╚══════════════════════════════════════════════════════════════╝
# 사용자가 USD에서 정리한 prim path.
# 72_ 핵심: 이제 child를 선택하지 않는다. 실제 좌표/정지/흡착/이동 기준은 OriBox root이다.
# 전제 USD 구조:
#   /World/OriBoxA_01 또는 /World/OriBoxB_01 = Rigid Body 있음, 실제 물리 박스 대표 좌표
#   하위 Small_Cardboard_box = Collider만 있음, Rigid Body 없음, local translate=(0,0,0)
BOX_PRIM_PATH = "/World/OriBoxA_01"
STOP_CHECK_PRIM_PATH = BOX_PRIM_PATH
BOX_ROOT_PATH = BOX_PRIM_PATH
BOX_MOVE_PRIM_PATH = BOX_ROOT_PATH
ORI_BOX_USD_PATH = _resolve_first_existing_path([PROJECT_DIR / "oriB" / "OriBoxB.usda", PROJECT_DIR / "oriA" / "OriBoxA.usda"], "OriBox USD")

# 사진에 보이는 값. 이 값은 "참고/생성 fallback"이다.
# 실제 흡착 판정은 이 숫자를 맹신하지 않고 root BOX_PRIM_PATH의 월드 bbox를 매 프레임 계산한다.
BOX_SCREENSHOT_LOCAL_TRANSLATE = np.array([1.92428, -4.20444, -0.41169], dtype=float)
BOX_SCREENSHOT_LOCAL_ROTATE_XYZ = np.array([0.23, 0.203, -0.001], dtype=float)
BOX_SCREENSHOT_LOCAL_SCALE = np.array([2.0, 2.0, 2.0], dtype=float)

# 흡착 판정 범위.
# 10_ 핵심: 큰 박스는 "윗면 중심점"이 아니라 "윗면 면적 안"으로 들어왔는지 봐야 한다.
# 기존 xy<=0.085 방식은 suction point가 박스 윗면 위에 있어도 중심에서 8.5cm 이상이면 실패했다.
# 로그상 suction point는 박스 윗면 안쪽에 들어왔지만 xy=0.09~0.10 정도라 계속 gate=False가 났다.
BOX_ATTACH_XY_TOL = 0.085       # 이전 중심점 방식 로그용 값. 10_에서는 판정 핵심으로 쓰지 않는다.
BOX_ATTACH_Z_MIN = -0.004       # 55_: 실제 윗면 흡착. 윗면을 살짝 파고드는 오차만 허용.
BOX_ATTACH_Z_MAX = 0.018        # 55_: 윗면보다 2.6cm 이내일 때만 흡착. 10cm 위 흡착 금지.
BOX_ATTACH_DIST_TOL = 0.080     # 55_: 로그/보조용. 실제 판정은 9점 top rectangle + z_gap.
BOX_TOP_SURFACE_MARGIN_X = 0.030  # 18_: 가장자리/옆면 흡착 방지. X 가장자리 3cm 제외.
BOX_TOP_SURFACE_MARGIN_Y = 0.055  # 18_: 옆면 흡착 방지. Y 가장자리 5.5cm 제외.
BOX_ATTACH_ON_TOP_SURFACE = True
PHYSICS_ATTACH_TOP_SURFACE_EPS = 0.002  # 55_: FixedJoint anchor는 흡착점 평균 높이가 아니라 실제 박스 윗면 z+3mm에 둔다.
# 흡착 실패로 계속 하강해 박스를 관통하는 것을 막기 위한 안전 정지.
PAUSE_IF_SUCTION_PENETRATES_WITHOUT_ATTACH = True
SUCTION_PENETRATION_STOP_Z_GAP = -0.030  # 18_: 윗면 기준 3cm 이상 파고들면 바로 정지.
# 11_ debug: scripted suction이 실제로 박스 bbox를 움직였는지 매 프레임 확인한다.
DEBUG_MOVE_PARENT_AND_FOLLOW = True
FOLLOW_ERROR_WARN_TOL = 0.030
MOTION_PROBE_DELTA_Z = 0.030

# 목표 위치는 박스 현재 위치 기준 상대 이동으로 잡는다.
# 예: [0.0, -0.45, 0.0]이면 현재 박스를 Y- 방향으로 45cm 옮겨 내려놓는다.
# 21_: 목표 위치는 "로봇 중심을 기준으로 박스 현재 위치의 반대편"으로 자동 계산한다.
# 예: robot_xy를 기준으로 box_xy를 대칭 이동한 좌표에 놓는다.
GOAL_MODE = "MIRROR_ACROSS_ROBOT_CENTER"
GOAL_OFFSET_FROM_BOX_CENTER = np.array([0.0, -0.45, 0.0], dtype=float)  # fallback: robot center를 못 읽을 때만 사용
GOAL_Z_OFFSET_FROM_PICK_CENTER = 0.280  # 23_: 반대편으로 이동할 때 바닥 높이가 아니라 들어올린 높이를 목표로 둔다.
PLACE_RELEASE_XY_TOL = 0.110          # 23_: XY 기준으로 반대편 좌표 근처에 오면 release
PLACE_RELEASE_Z_TOL = 999.0           # 23_: release 판정에서 Z는 직접 쓰지 않는다. 하강을 기다리면 박스가 끌려 내려감

# 15_ 핵심: 박스 USD 원점/루트가 바닥 기준이어도 bbox로 실제 윗면을 계산한다.
# 로봇 pick 목표는 박스 중심이 아니라 top_center + clearance로 올린다.
USE_BOX_TOP_CENTER_AS_PICK_TARGET = True
BOX_PICK_TOP_CLEARANCE = 0.090  # 18_: 옆면이 아니라 윗면에서 잡도록 pick 목표 높이를 더 올림.

# 16_ 핵심: 네가 M0609/발판 위치를 다시 조정한 뒤 로그상 suction point가
# 박스 윗면 기준으로 X는 +7cm, Y는 -25cm 정도 빗나갔다.
# controller에 넣는 picking_position을 반대로 보정해서 VGC10 흡착점이 박스 윗면 중앙으로 오게 한다.
# 로그 기준: dx=suction_x-box_top_x≈+0.07, dy≈-0.25 → pick target 보정=[-0.07,+0.25,0]
PICK_TARGET_MANUAL_OFFSET = np.array([0.0, 0.0, 0.0], dtype=float)  # 53_: manual offset 대신 pre-align에서 box_top 좌표로 직접 정렬

# 흡착이 되면 place까지 가지 않고, 박스를 붙인 상태로 로봇을 초기 자세(0 joint, 일자로 뻗은 상태)로 복귀한다.
RETURN_HOME_IMMEDIATELY_AFTER_ATTACH = False
KEEP_BOX_ATTACHED_DURING_HOME_RETURN = False

# home 복귀 목표. M0609 joint 수가 6이므로 기본은 0 자세.
HOME_TARGET_JOINTS_CONFIG = None  # None이면 np.zeros_like(current_joints) 사용

# release 후 로봇 복귀
RETURN_HOME_AFTER_RELEASE = True
HOME_RETURN_STEPS = 160

# 19_ 핵심: 흡착 후 박스를 목표 위치로 텔레포트/강제 배치하지 않는다.
# VGC10이 visual-only라서 흡착 중에는 scripted carry가 필요하지만,
# 일정 높이만 들어올리면 바로 release하고 박스 물리를 다시 켠 뒤 로봇만 홈으로 복귀한다.
# 21_: 20_의 핵심은 유지한다. 잡자마자 바로 옆으로 끌지 않고 먼저 들어 올린다.
# 다만 20_처럼 수직 상승 후 바로 release하지 않고, 반대편 goal 근처에서 release한다.
RELEASE_AFTER_LIFT_DELTA_Z = 0.060
RELEASE_AFTER_ATTACH_MIN_STEPS = 8
RELEASE_AT_CURRENT_POSE = True  # goal 근처에 도달했을 때 현재 pose에서 흡착 해제. 시작/스폰 위치 강제 이동은 하지 않음.

# 23_ 핵심: 흡착 후에는 PickPlaceController event를 더 진행하지 않고,
# RMPFlow로 suction point 목표를 직접 나눈다.
# 흐름: 위로 들어올림 -> 높은 위치에서 로봇 중심 반대편 XY로 이동 -> 목표 위에서 살짝 하강 -> release -> robot home.
CUSTOM_CARRY_AFTER_ATTACH = True
CUSTOM_CARRY_SAFE_SUCTION_Z = 1.18          # 51_: 9점 물리 흡착 테스트에서도 안전 높이 기준으로 사용
CUSTOM_LIFT_DELTA_Z = 0.16                  # 49_: 너무 높게 들지 않고 BoxAprop 위로 이동 가능한 높이까지만 상승.
CUSTOM_LOWER_CLEARANCE_Z = 0.010            # release 직전 박스 윗면과 suction 사이 여유. 음수면 눌러 들어감.
CUSTOM_LIFT_TOL = 0.060  # 49_: 1.42m 근처까지 오면 다음 phase로 넘어가게 완화
CUSTOM_MOVE_XY_TOL = 0.080  # move phase 완료 기준. 너무 작으면 이동 phase가 max step으로만 넘어간다.
CUSTOM_LOWER_TOL = 0.035
CUSTOM_PHASE_MAX_STEPS = 900  # 50_: slot까지 자연스럽게 더 접근할 시간을 준다
CUSTOM_RELEASE_AFTER_LOWER_MIN_STEPS = 20
CUSTOM_LOG_INTERVAL = 55  # 128_: 불필요한 custom phase 로그 축소

# 49_ 핵심: BoxAprop 적재에서 로봇이 slot XY에 완전히 못 붙으면 lower phase가 무한 대기했다.
# lower 진입은 조금 더 넓게 허용하고, 그래도 못 내려가면 timeout 후 slot 중심으로 snap-release한다.
STACK_DIRECT_LOWER_ALLOW_XY_TOL = 0.120
STACK_RELEASE_ON_LOWER_TIMEOUT = True  # 50_: timeout이면 현재 위치에서 release, snap은 하지 않음


# 53_ 핵심: 작업영역에 들어온 박스의 bbox top_center 좌표를 읽은 뒤,
# PickPlaceController로 대략 접근만 하는 대신 suction 중심을 box_top 위로 직접 정렬한다.
# 로그에서 dx≈0.27m로 빗나간 문제가 있었기 때문에, 흡착 판정 전에 XY 정렬을 먼저 끝낸다.
PRE_ATTACH_ALIGN_ENABLED = True
PRE_ATTACH_ALIGN_XY_TOL = 0.030          # suction 중심과 box_top 중심의 XY 허용 오차
PRE_ATTACH_ALIGN_Z_TOL = 0.040           # 55_: XY 정렬 단계에서 높이 허용 오차
PRE_ATTACH_ALIGN_HEIGHT = 0.080          # 55_: XY 정렬은 box_top보다 8cm 위에서 먼저 수행
PRE_ATTACH_CONTACT_GAP = 0.010           # 55_: 실제 윗면 접촉용. box_top보다 1.8cm 위까지 접근.
PRE_ATTACH_MIN_STEPS_BEFORE_ATTACH = 8   # 너무 빠른 오검출 방지
PRE_ATTACH_MAX_STEPS = 900               # 여기까지 안 맞으면 home return 후 재시도
PRE_ATTACH_LOG_INTERVAL = 10
PRE_ATTACH_EVALUATE_GRID_EVERY_STEP = True

# 55_ 핵심 수정:
# 53_에서는 pre-align 시작 순간의 EE 방향을 고정해서, VGC10/흡착점이 하늘을 보는 자세로
# 위치 이동을 시도했다. 그 결과 box_top 좌표는 읽지만 suction XY가 계속 빗나갔다.
# 이번 버전은 pre-align 동안 orientation constraint를 빼고 위치를 먼저 맞춘다.
# 즉, 흡착 전에는 '하늘 보는 방향 고정'을 하지 않고 box_top 위 좌표로 suction 중심을 맞춘다.
PRE_ATTACH_KEEP_CURRENT_ORIENTATION = False  # 109_: pre-align은 위치 우선, attach 후 carry에서 orientation lock

# 55_: 시각/판정용 9점 흡착 그리드는 실제 USD prim의 기울어진 local pose 대신
# 현재 suction 중심 기준 world XY 평면에 가상으로 만든다.
# 목적: 상자 윗면 판정이 gripper가 하늘을 보거나 기울어진 자세에 끌려가지 않게 하기 위함.
SUCTION_GRID_EVALUATE_AS_WORLD_XY_GRID = True
SUCTION_GRID_WORLD_MARKER_ROOT_PATH = "/World/vgc10_world_xy_suction_grid_debug"

# 완전 대칭 좌표가 로봇 도달 범위를 벗어나면 박스 시작점 기준 이동 거리를 제한한다.
# 목표 좌표가 너무 멀면 goal_xy_error가 계속 1m 이상으로 남고 release가 안 되므로 안전장치로 둔다.
MIRROR_GOAL_MAX_XY_DISTANCE_FROM_PICK = 0.65
ROBOT_CENTER_CANDIDATE_PATHS = [
    # 123_: 새 USD에서는 /World/Cube가 로봇 중심 기준이 아니어서 goal_angle이 -6도만 계산됨.
    # goal_angle을 다시 쓸 때도 실제 1번 로봇 기준으로 계산되도록 Cube를 후보에서 제외한다.
    ROBOT_PRIM_PATH + "/base_link",
    ROBOT_PRIM_PATH,
    ACTIVE_ROBOT_ROOT_PATH,
]

# 24_ 핵심: 로봇/발판 큐브를 가로질러 직선으로 이동하면 박스와 팔이 관통처럼 보인다.
# 실제 충돌 회피가 아니라 RMPFlow 목표점을 여러 waypoint로 나누어 발판 주변을 돌아가게 만든다.
CARRY_AVOID_BASE_AND_ARM = False  # 49_: 3개 적재에서는 우회 waypoint가 오히려 멀어져서 비활성화
CARRY_FORBIDDEN_RADIUS_XY = 0.58        # 로봇 중심 주변 금지 반경. 큐브+로봇팔 주변을 대략 피한다.
CARRY_ROUTE_MARGIN_XY = 0.24            # 금지 반경 바깥으로 추가 여유.
CARRY_ROUTE_SIDE_SIGN = 1.0             # +1 또는 -1. 경로가 이상하면 -1로 바꿔 반대쪽으로 돌아가게 한다.
CUSTOM_MIN_SUCTION_Z_FOR_XY_MOVE = 0.0 # 59_: move_stack 중 target_suction이 현재 위치로 튀는 guard 완전 비활성화
CUSTOM_LOWER_ONLY_AT_GOAL = True        # 목표 XY 근처에 도착하기 전에는 절대 하강하지 않는다.


# 28_ 핵심: 벽에 닿아 멈춘 박스를 흡착한 뒤, 먼저 위로만 빼내되 27_보다 lift 대기 시간을 줄인다.
# 박스가 벽 높이/베이스 주변보다 충분히 올라가기 전까지 joint_1 회전을 금지한다.
# 흐름: 흡착 -> joint_2/3 수직 lift 우선 -> 높이 조건 만족 후 joint_1 반 바퀴 회전 -> 살짝 내림 -> release.
# link_1/link_2의 transform을 직접 수정하지 않는다. articulation에서는 joint target을 줘야 한다.
# 47_ 핵심: 쌓기 목표 좌표는 BoxAprop 위의 슬롯 좌표이므로
# joint swing(반 바퀴 회전) 대신 Cartesian waypoint가 task.goal_center를 따라가게 한다.
# 109_: 상자를 기울이지 않고 수직으로 들기 위해 관절 delta(JOINT_SWING)가 아니라
# RMPFlow/IK에 suction position + fixed orientation을 함께 준다.
# attach 순간의 EE orientation을 유지하므로 link_6가 기울면서 박스를 같이 기울이는 문제를 줄인다.

# =============================================================================
# 170_: 164 안정판 기반 release 직전 APalt pose 진단 전용 버전.
#       흡착/운반/release 흐름은 변경하지 않고 로그만 추가.
# =============================================================================
# 122_ / 117_ SIMPLE 3-STEP MODE - JOINT_1 ROTATE
# 목적: 복잡한 boxaprop/path_move/자동 다음 박스 반복 제거.
# 흐름: (1) 수직 상승 -> (2) joint_1/link_1 회전으로 뒤쪽 큐브 방향 이동 -> (3) 같은 높이만큼 역방향 하강 후 release.
# 주의: 114_에서 joint_2를 돌리면 팔을 접는 동작이 되어 상자가 뒤쪽으로 가지 않는다.
#      117_는 사용자가 의도한 “뒤쪽으로 보내는 최소 회전”을 joint_1 회전으로 해석한다.
# =============================================================================
CUSTOM_CARRY_MODE = "VERTICAL_JOINT1_REVERSE"  # 122_: 117 동작 유지. 117_: 3단계: 수직 상승 -> joint_1 회전 -> 역방향 하강
# 80_: 테스트 목적은 "물리 적용된 박스가 joint로 붙어서 같이 들리는지" 확인하는 것.
#      팔레타이징 이동/스윙은 일단 빼고, 흡착 -> FixedJoint -> 수직 lift -> release만 한다.
PHYSICS_FIXED_JOINT_LIFT_ONLY_TEST = False  # 108_: lift only가 아니라 lift -> swing -> release까지 진행
PHYSICS_LIFT_KEEP_ATTACHED_AND_PAUSE = False  # 108_: 들어올린 뒤 pause하지 않고 joint_1 swing 후 release까지 진행
PHYSICS_FIXED_JOINT_STABILIZE_STEPS = 55  # legacy 값. 103_ 진단 모드에서는 아래 DIAGNOSTIC_HOLD_STEPS를 사용한다.
PHYSICS_FIXED_JOINT_DIAGNOSTIC_NO_LIFT = False
PHYSICS_DIAGNOSTIC_HOLD_STEPS = 120

# 104_: 원본 USD를 저장하지 않고 실행 중에만 박스 MassAPI / inertia를 명시해 검증한다.
PHYSICS_MASS_INERTIA_DIAG_ENABLED = True
PHYSICS_MASS_INERTIA_DIAG_MASS = 2.0
PHYSICS_MASS_INERTIA_DIAG_USE_BBOX_COM = True
PHYSICS_MASS_INERTIA_DIAG_INERTIA_SCALE = 1.0
PHYSICS_DIAGNOSTIC_SAMPLE_STEPS = {0, 1, 2, 3, 5, 10, 20, 30, 40, 60, 80, 100, 120}
PHYSICS_DIAGNOSTIC_LOG_COLLIDER_TREE = True
PHYSICS_DIAGNOSTIC_LOG_EVERY_STEP_UNTIL = 5
JOINT_CARRY_LOG_INTERVAL = 10
TOP_LOCK_FOLLOW_DESIRED_PATH = False  # 75_: joint_1 회전 중에는 박스가 실제 suction을 따라가게 둔다.
TOP_LOCK_PATH_LIFT_STEPS = 150
TOP_LOCK_PATH_MOVE_STEPS = 1
TOP_LOCK_PATH_LOWER_STEPS = 1
TOP_LOCK_PATH_SETTLE_STEPS = 80

# 59_: 다시 한 단계 낮춘 테스트.
# 목표는 적재가 아니라 “상자 윗면 중심 흡착 → 실제 suction 위치를 따라 수직으로만 들어올리기” 검증이다.
VERTICAL_LIFT_ONLY_TEST = True
VERTICAL_LIFT_DELTA_Z = 0.24  # 109_: RMPFlow fixed-orientation으로 수직 lift 높이
VERTICAL_LIFT_HOLD_STEPS = 130  # 81_: 들어올린 상태 확인 시간
VERTICAL_LIFT_PAUSE_AFTER_SUCCESS = False
# 59_: 수직 리프트 후 BoxAprop 위 좌표까지 회전/이동만 테스트한다. release/stack은 하지 않는다.
VERTICAL_LIFT_THEN_CUBE_OVER_ENABLED = False  # 114_: boxaprop/path_move 사용 금지. 3단계만 수행.  # 109_: 수직 lift 후 Cube/APalt 쪽 이동/하강까지 테스트
CUBE_OVER_MOVE_STEPS = 380  # 109_: orientation 유지한 상태로 천천히 이동
CUBE_OVER_HOLD_STEPS = 40
BOXAPROP_LOWER_STEPS = 170  # 109_: 기울임/튀김 방지용 저속 하강
BOXAPROP_SETTLE_STEPS = 70
BOXAPROP_RELEASE_ON_SETTLE = True

# 117_: 사용자가 원하는 흐름
#   1) 흡착 후 최대한 수직으로 상승: RMPFlow Cartesian Z lift + fixed EE orientation
#   2) 뒤쪽 큐브 방향으로 최소 움직임: joint_1/link_1만 부드럽게 회전
#   3) 내려놓기: 1)의 반대 방향으로 현재 XY에서 수직 하강
# 이 모드는 109/110처럼 목표 XY까지 RMPFlow로 길게 끌고 가지 않는다.
# 따라서 이상한 우회/큰 움직임을 줄이고, “뒤쪽으로 보내는 회전”을 joint_1로 확인한다.
# 124_: 현재 파일은 마지막 위치 회전은 신경쓰지 않고 drop release 확인이 우선이므로 manual_delta 유지.
# APalt와 상자 사이 XY 위치가 맞지 않으면, 이 manual 회전각을 먼저 조절한다.
HYBRID_JOINT1_ROTATE_MODE = "manual_delta"  # goal_angle 대신 수동 회전량 사용
HYBRID_JOINT1_ROTATE_SIGN = 1.0  # 회전 방향이 반대면 -1.0으로 변경
HYBRID_JOINT1_ROTATE_FALLBACK_DEG = 160.0  # joint_1 실제 회전각. APalt보다 덜 가면 +, 지나치면 -
HYBRID_JOINT1_ROTATE_MAX_DEG = 360.0  # 117_: APalt 방향까지 더 회전 허용  # 117_: 115번 120도에서 뒤쪽까지 부족해서 160도까지 허용
HYBRID_JOINT1_ROTATE_STEPS = 240  # 117_: 회전량 증가에 맞춰 더 천천히 수행  # 117_: 회전량 증가에 맞춰 더 천천히 수행
HYBRID_REVERSE_LOWER_STEPS = 170  # 들어올린 높이만큼 반대로 내려놓기.
HYBRID_SETTLE_STEPS = 30  # 내려놓은 뒤 짧게 안정화.
HYBRID_RELEASE_ON_SETTLE = True

# 145_: n개 적재용 회전 패턴. 홀수 번째 박스는 160도, 짝수 번째 박스는 130도.
# 기준: 1번=왼쪽, 2번=오른쪽, 3번=왼쪽 위, 4번=오른쪽 위.
STACK_JOINT1_DEG_PATTERN_145 = (220.0, 155.0, 220.0, 155.0)
BOX_STACK_HEIGHT_145 = 0.24        # 3개 이상 적재할 때 APalt를 내릴 높이. 상자 높이에 맞춰 수정
APALT_LOWER_LOG_INTERVAL_145 = 15  # APalt lowering 중 박스 위치 로그 간격
APALT_LOWER_SPEED_NOTE_145 = "STACK_LOWER_SUPPORT_STEPS 값을 키우면 더 천천히 내려갑니다"


# =============================================================================
# 155_: APalt 정답지 큐브 기반 hybrid palletizing
# 목적:
# - APalt 위에 사용자가 직접 둔 정답지 큐브 위치를 최종 place 기준으로 사용한다.
# - 큰 이동은 기존 link_1 회전 장점을 유지하고, 마지막만 slot marker 좌표로 보정한다.
# - 컨베이어 위 다른 상자보다 충분히 높게 든 뒤 joint_1 회전을 시작한다.
# 정답지 큐브 prim path:
#   /World/APalt_slot_01
#   /World/APalt_slot_02
# 주의: 정답지 큐브는 위치/방향 기준점용이다. Rigid Body/Collider는 없어도 된다.
# =============================================================================
SLOT_MARKER_PALLETIZING_ENABLED_155 = True
SLOT_MARKER_PATHS_155 = (
    "/World/APalt_slot_01",
    "/World/APalt_slot_02",
)
# slot marker가 2개뿐일 때 3번째 박스부터 1/2번을 반복할지 여부.
SLOT_MARKER_WRAP_IF_SHORT_155 = True
# marker 중심이 실제 박스 center 목표라고 본다. 즉 marker를 박스와 같은 크기/높이로 APalt 위에 올려둔다.
SLOT_MARKER_USE_BBOX_CENTER_155 = True
# link_1 회전 후 marker 위로 갈 때, 바로 하강하지 않고 marker 위쪽에서 접근할 높이 여유.
SLOT_MARKER_APPROACH_CLEARANCE_Z_155 = 0.06  # 160_: 불필요한 재상승을 줄임. 실제 no-up은 runtime에서 한 번 더 보정
# marker 위 마지막 보정 이동 step. 너무 작으면 팔이 꺾일 수 있어 180~260 권장.
SLOT_MARKER_FINAL_MOVE_STEPS_155 = 220
SLOT_MARKER_FINAL_LOWER_STEPS_155 = 150
SLOT_MARKER_FINAL_SETTLE_STEPS_155 = 35
# 컨베이어 위 다른 OriBoxA_* 최고 높이보다 이만큼 더 높게 들어올린 뒤 swing 시작.
CONVEYOR_SAFE_LIFT_ENABLED_155 = True
CONVEYOR_SAFE_LIFT_CLEARANCE_Z_155 = 0.22
CONVEYOR_SAFE_LIFT_INCLUDE_PREFIXES_155 = ("/World/OriBoxA_",)
CONVEYOR_SAFE_LIFT_EXCLUDE_ATTACHED_155 = False  # False면 현재 잡은 박스도 높이 계산에 포함. 최고 높이 기준이라 안전함.
# 방향 정렬은 다음 단계에서 안정화한다. 이번 155는 marker 위치 기반 place를 먼저 확인한다.
SLOT_MARKER_YAW_ALIGN_ENABLED_155 = False

# 156_: joint_1 회전 기준 변경.
# 기존 155는 "흡착한 현재 joint_1 + 220도" 방식이라, 집는 위치가 바뀌면 최종 회전각도 같이 바뀌었다.
# 이번 버전은 "로봇 초기화 상태 joint_1 + 목표 각도"를 절대 목표로 사용한다.
# 예: 초기 기준에서 -10도 지점에서 집었고 목표가 +220도라면, 현재 위치에서는 +230도 이동해 최종 +220도에 도달한다.
ABSOLUTE_JOINT1_FROM_INITIAL_ENABLED_156 = True
ABSOLUTE_JOINT1_LOG_156 = True
ROBOT_INITIAL_JOINTS_156 = None

# =============================================================================
# 157_: APalt_slot 정답지 큐브 기반 joint_1 자동 회전량 계산
# 목적:
# - 220/155처럼 손으로 쓴 회전각을 쓰지 않는다.
# - 로봇 base 기준으로 현재 박스 방향(x3)과 목표 slot 방향(x1/x2)을 계산한다.
# - 현재 자세에서 필요한 회전량은 (slot 방향 - 현재 박스 방향)이다.
# - 최종 정확도는 기존처럼 APalt_slot 좌표 보정(move_over/lower/settle)이 담당한다.
# =============================================================================
AUTO_JOINT1_FROM_SLOT_MARKER_ENABLED_157 = True
AUTO_JOINT1_USE_SHORTEST_DELTA_157 = True  # True: -180~+180도 중 짧은 방향으로 회전
AUTO_JOINT1_MAX_ABS_DELTA_DEG_157 = 170.0  # 159에서 slot별 방향 강제 시 AUTO_JOINT1_MAX_ABS_DELTA_DEG_159가 우선
AUTO_JOINT1_MIN_DELTA_DEG_157 = 0.0        # 너무 작은 회전도 허용. 필요하면 5~10도 설정
AUTO_JOINT1_LOG_157 = True

# 158_: joint_1 회전 후 다음 Cartesian 보정 phase가
#       회전 전 lift_suction을 start로 다시 사용하면서 로봇이 되돌아가는 문제를 막는다.
#       slot_marker_move_over/lower/settle phase가 시작될 때 실제 현재 suction 위치를 start로 재설정한다.
PHASE_DYNAMIC_START_AFTER_JOINT1_ENABLED_158 = True
PHASE_DYNAMIC_START_NAMES_158 = (
    "slot_marker_move_over_155",
    "slot_marker_lower_155",
    "slot_marker_settle_155",
)
PHASE_DYNAMIC_START_LOG_158 = True


# =============================================================================
# 159_: slot별 회전 방향 분리 + 정답지 큐브 yaw 정렬
# 목적:
# - APalt_slot_01/02가 서로 다른 쪽으로 접근하게 하여, 이미 놓은 상자를 지나가며 치는 문제를 줄인다.
# - 상자가 컨베이어에서 가로/세로/애매한 방향으로 들어와도 release 직전에 slot marker와 같은 yaw로 맞춘다.
# =============================================================================
SLOT_ROUTE_SIGN_BY_MARKER_159 = (-1.0, +1.0)   # slot_01=음(-)방향, slot_02=양(+)방향. 3/4번은 1/2 반복.
SLOT_ROUTE_SIGN_FORCE_159 = True               # True면 shortest delta 대신 slot별 지정 방향으로 회전한다.
AUTO_JOINT1_MAX_ABS_DELTA_DEG_159 = 260.0      # slot별 방향을 강제하면 180도를 넘을 수 있으므로 157의 170 제한보다 넓게 허용.
SLOT_MARKER_YAW_ALIGN_ENABLED_159 = False  # 160_: release 직전 순간 yaw 보정 금지. 로봇 동작 검증을 위해 snap/teleport 제거.
SLOT_MARKER_YAW_ALIGN_LOG_159 = True
SLOT_MARKER_YAW_ALIGN_ZERO_VEL_159 = True

# 163_: 160 안정판 기반. 흡착 후 joint_6 yaw 보정/pose recenter는 제거한다.
# 164_: 강제 보정 없이 box/slot local X,Y축 yaw 진단을 확장하고, 2번째 release 직후 정지한다.
# 161/162에서 확인된 문제: 흡착 후 wrist 회전은 yaw는 줄여도 box center를 크게 밀었다.
# 이번 버전은 상자/slot yaw를 흡착 전과 release 전에 기록하되, 상자를 순간 회전시키지 않는다.
PRE_PICK_YAW_DIAGNOSTIC_ENABLED_163 = True
PRE_PICK_YAW_OK_TOL_DEG_163 = 5.0
PRE_PICK_YAW_WARN_TOL_DEG_163 = 15.0
POST_PICK_YAW_CORRECTION_DISABLED_163 = True

# =============================================================================
# 171_: release 직전 APalt_slot 정답지 방향 기준 RMPFlow yaw align
# - 170에서 확인된 상태: center_xy_err=0, level OK, yaw만 18.8도 틀어짐.
# - pre-align / suction / link_1 회전 / slot_marker move/lower/settle 흐름은 건드리지 않는다.
# - slot_marker_settle_155 이후, release 직전에만 현재 suction 위치를 유지하고
#   EE target orientation을 APalt yaw 오차만큼 짧게 보정한다.
# - box transform 직접 회전, snap, teleport, joint_6 단독 trim 금지.
# =============================================================================
PRE_RELEASE_YAW_ALIGN_RMPFLOW_ENABLED_171 = True
PRE_RELEASE_YAW_ALIGN_ONLY_IF_CENTER_LEVEL_OK_171 = True
PRE_RELEASE_YAW_ALIGN_STEPS_171 = 160  # 173_: 172 probe에서 검증된 것처럼 충분한 step 동안 실제 RMPFlow action 적용
PRE_RELEASE_YAW_ALIGN_MAX_DEG_171 = 25.0
PRE_RELEASE_YAW_ALIGN_MIN_DEG_171 = 2.0
PRE_RELEASE_YAW_ALIGN_CENTER_ABORT_M_171 = 0.018
PRE_RELEASE_YAW_ALIGN_LEVEL_ABORT_DEG_171 = 3.0
PRE_RELEASE_YAW_ALIGN_LOG_INTERVAL_171 = 20
PRE_RELEASE_YAW_ALIGN_KEEP_SUCTION_POSITION_171 = True
PRE_RELEASE_YAW_ALIGN_INSERT_AFTER_SETTLE_171 = True
PRE_RELEASE_YAW_ALIGN_ABORT_PAUSE_171 = True
PRE_RELEASE_YAW_ALIGN_REQUIRE_AXIS_OK_BEFORE_RELEASE_171 = True  # 173_: yaw 보정 실패 시 release 차단

# =============================================================================
# 184_: 2번째 상자 정렬 시간 단축
# 목적:
# - 기존 173은 slot_marker_settle_155 이후 release 직전에 pre_release_yaw_align_171을
#   별도 160 step 동안 수행했다.
# - 2번째 상자는 로그상 yaw 오차가 약 18.8deg이며, 별도 정렬 phase 55 step 안에 이미
#   거의 0deg까지 줄어든다. 따라서 이 yaw 정렬을 내려가는 slot_marker_lower_155 / settle
#   과정에 흡수해 별도 대기 시간을 제거한다.
# - 박스 transform 직접 회전/snap/teleport는 여전히 금지한다. RMPFlow target orientation만
#   lower/settle 동안 함께 바꾼다.
# =============================================================================
FUSED_SECOND_BOX_YAW_ALIGN_ENABLED_184 = True
FUSED_YAW_ALIGN_ALL_FOUR_NOTE_185 = "185_: 1~4번째 모두 별도 pre_release_yaw_align phase 없이 lower/settle 중 yaw 정렬"
FUSED_YAW_ALIGN_SLOT_INDICES_184 = (0, 1, 2, 3)  # 185_: zero-based. 1~4번째 상자 모두 lower/settle 중 yaw 정렬
FUSED_YAW_ALIGN_PHASE_NAMES_184 = ("slot_marker_lower_155", "slot_marker_settle_155")
FUSED_YAW_ALIGN_START_PHASE_184 = "slot_marker_lower_155"
FUSED_YAW_ALIGN_DISABLE_FINAL_PRE_RELEASE_PHASE_184 = True
FUSED_YAW_ALIGN_REUSE_IN_SETTLE_184 = True
FUSED_YAW_ALIGN_MIN_DEG_184 = 2.0
FUSED_YAW_ALIGN_MAX_DEG_184 = 25.0
FUSED_YAW_ALIGN_CENTER_TOL_M_184 = 0.080  # lower 시작 전 move_over 후에는 이 범위 안이면 orientation 보정 시작
FUSED_YAW_ALIGN_LEVEL_TOL_DEG_184 = 5.0
FUSED_YAW_ALIGN_LOG_184 = True


# 164_: 강제 보정 없이 진단만 확장.
# - box local X/Y축과 APalt_slot local X/Y축 yaw를 모두 찍는다.
# - 두 번째 상자 release 직후 정지해서 로그만 확인한다.
AXIS_YAW_DIAGNOSTIC_ENABLED_164 = True
AXIS_YAW_DIAG_LOG_BBOX_164 = True
DIAG_STOP_AFTER_RELEASE_ENABLED_164 = False  # 185_: 2번째 release 후 진단 정지 금지. FORKLIFT_TRIGGER_COUNT까지 계속 진행
DIAG_STOP_AFTER_RELEASE_COUNT_164 = 4  # 185_: 참고값. 실제 정지는 위 enabled=False라 사용하지 않음
DIAG_STOP_AFTER_RELEASE_SKIP_APALT_LOWER_164 = True

# =============================================================================
# 160_: no-snap yaw + 빠른 place + release 후 안정화 대기
# 목적:
# - 상자를 APalt_slot yaw로 순간 이동/회전시키는 보정을 제거한다.
# - yaw는 실제 로봇 경로/자세가 맞추지 못하면 로그상 실패로 남긴다.
# - joint_1 회전 후 이미 slot 근처에 있으면 move_over에서 다시 위로 살짝 들어올리지 않는다.
# - 2번째 release 직후 바로 APalt를 내리지 않고 1~2초 물리 안정화 시간을 준다.
# =============================================================================
SLOT_MARKER_YAW_SNAP_DISABLED_160 = True
SLOT_MARKER_YAW_CHECK_LOG_160 = True
SLOT_MARKER_NO_UP_BEFORE_RELEASE_160 = True
SLOT_MARKER_SKIP_MOVE_IF_NEAR_XY_160 = True
SLOT_MARKER_SKIP_MOVE_XY_TOL_160 = 0.08
POST_RELEASE_SETTLE_BEFORE_APALT_LOWER_ENABLED_160 = True
POST_RELEASE_SETTLE_BEFORE_APALT_LOWER_STEPS_160 = 120
POST_RELEASE_SETTLE_LOG_INTERVAL_160 = 30

def get_stack_joint1_deg_145(slot_index):
    """0-based slot_index 기준으로 joint_1 회전각을 반환한다.
    slot 0/2 = 160도, slot 1/3 = 130도.
    """
    try:
        idx = int(slot_index)
    except Exception:
        idx = 0
    pattern = tuple(float(x) for x in STACK_JOINT1_DEG_PATTERN_145)
    if not pattern:
        return float(HYBRID_JOINT1_ROTATE_FALLBACK_DEG)
    return float(pattern[idx % len(pattern)])

# =============================================================================
# 124_ DROP RELEASE MODE
# 목적: 마지막 위치/회전 상태는 그대로 두고, 흡착 FixedJoint를 제거해서
#       물리 적용된 상자가 중력으로 떨어지는지 확인한다.
# 주의: link_1 Transform을 직접 돌리거나 박스를 순간이동시키지 않는다.
#       기존 117/123 흐름이 끝난 뒤 FixedJoint만 제거한다.
# =============================================================================
DROP_RELEASE_AFTER_SETTLE_124 = True
# 126_: VERTICAL_JOINT1_REVERSE 모드에서는 place_enabled가 False여도 마지막 settle 후 무조건 FixedJoint를 제거한다.
FORCE_DROP_RELEASE_AFTER_SETTLE_126 = True
# 126_: 떨어지는 장면 확인용. 0.10이면 10cm 더 높은 곳에서 흡착을 떼므로 낙하가 보인다.
#       0.00이면 APalt 위에 거의 올려진 상태에서 떼기 때문에 떨어지는 장면이 안 보일 수 있다.
DROP_RELEASE_EXTRA_Z_125 = 0.10
DROP_RELEASE_OBSERVE_STEPS_124 = 0  # 163_: release 후 바로 home return 시작      # 128_: release 후 관찰 step. 핵심 진단 로그만 출력
DROP_RELEASE_LOG_INTERVAL_124 = 60        # 128_: drop 관찰 로그 간격 축소
DROP_RELEASE_RETURN_HOME_AFTER_OBSERVE_124 = True   # 127_: release 후 home 복귀까지 실행
DROP_RELEASE_ZERO_VELOCITY_BEFORE_DETACH_124 = False # True면 떼는 순간 속도 제거 후 순수 중력 낙하

# 127_: 126에서 phase 완료 블록이 실행되지 않으면 release 코드에 못 들어가는 문제가 있었다.
# 그래서 상자가 pick 위치에서 충분히 이동했고, APalt 위쪽에서 일정 step 이상 정지/유지되면
# phase 상태와 무관하게 FixedJoint를 제거하는 안전장치(watchdog)를 둔다.
FORCE_DROP_RELEASE_WATCHDOG_127 = False  # 128_: 중간에 갑자기 흡착 해제될 수 있어 watchdog release 금지
FORCE_DROP_RELEASE_WATCHDOG_STEPS_127 = 30
FORCE_DROP_RELEASE_MIN_XY_MOVE_127 = 0.25      # pick 위치에서 XY로 이 이상 이동하면 최종 위치 근처로 판단
FORCE_DROP_RELEASE_MIN_Z_ABOVE_PICK_127 = 0.06 # pick 높이보다 이 이상 높으면 낙하 확인용 release 가능
FORCE_DROP_RELEASE_OBSERVE_STEPS_127 = 180
FORCE_DROP_RELEASE_LOG_INTERVAL_127 = 20
RETURN_HOME_AFTER_WATCHDOG_DROP_127 = True

# =============================================================================
# 128_ RELEASE DIAGNOSIS MODE
# 목적: 같은 증상이 반복되는 원인을 한 번에 분리한다.
# - 중간 이동 중에는 절대 release하지 않는다.
# - 마지막 settle 완료 지점에서만 release한다.
# - release 전/후 joint 잔존, rigid/kinematic/collision/gravity 상태를 짧게 출력한다.
# - release 직후 상자 root/subtree를 dynamic으로 복구해도 안 떨어지는지 확인한다.
# =============================================================================
RELEASE_DIAG_128_ENABLED = True
RELEASE_DIAG_SCAN_ALL_JOINTS_128 = True
RELEASE_DIAG_FORCE_DYNAMIC_AFTER_DETACH_128 = True
RELEASE_DIAG_RESTORE_COLLISION_128 = True
RELEASE_DIAG_ZERO_VELOCITY_128 = True
RELEASE_DIAG_MAX_ROWS_128 = 24
RELEASE_DIAG_OBSERVE_SAMPLES_128 = (0, 30, 60, 120, 180, 240, 299)
RELEASE_DIAG_CLEAR_DROP_TRACKER_128 = True

# 114_ 호환용 이름. 다른 코드 블록이 이 이름을 참조해도 실행되도록 유지한다.
HYBRID_LINK2_ROTATE_TARGET_DEG = -125.0
HYBRID_LINK2_ROTATE_MIN_DEG = -135.0
HYBRID_LINK2_ROTATE_STEPS = HYBRID_JOINT1_ROTATE_STEPS

# 62_: release 순간에는 PhysX/USD physics 속성 변경 금지. 현재 pose에 그대로 놓고 attach 플래그만 OFF.
BOXAPROP_SAFE_RELEASE_NO_PHYSICS_TOGGLE = True
BOXAPROP_RELEASE_MAX_SUCTION_ERR = 0.65  # 109_: RMPFlow 목표 오차 허용을 조금 넓힘  # 이보다 크면 내려놓기 실패로 보고 release하지 않음
# 67_: BoxAprop 목표로 못 가는데 step timeout만으로 lower/release까지 진행하면
#      두 번째 박스가 엉뚱한 위치에서 release abort로 멈춘다.
#      move/lower/settle 단계에서 목표 오차가 너무 크면 pause하지 않고 실패 처리 후 home 복귀한다.
BOXAPROP_ABORT_HOME_ON_TARGET_ERROR = False  # 109_: 첫 upright RMPFlow 테스트에서는 목표 오차가 있어도 로그 확인 우선
BOXAPROP_MOVE_MAX_SUCTION_ERR = 0.45
BOXAPROP_LOWER_MAX_SUCTION_ERR = 0.45
CUBE_OVER_HEIGHT_MARGIN = 0.000  # lift_suction 높이를 그대로 유지. 더 높게 지나가려면 0.03~0.05 추가.
CUBE_OVER_MAX_TARGET_ERR_WARN = 0.45
VERTICAL_LIFT_MAX_SUCTION_TARGET_ERR = 0.090
JOINT_LIFT_STEPS = 240  # 102_: 안정화 후 더 천천히, 79보다 조금 더 높게 물리 lift 확인
JOINT_SWING_STEPS = 200  # 108_: joint_1 swing을 천천히 수행
JOINT_LOWER_STEPS = 1  # 75_: joint_1 회전 후 j2/j3 lowering 금지. 급격한 꺾임/XY 튐 방지
JOINT_SETTLE_STEPS = 1  # 75_: release 전 대기 최소화

# 들어올리기 관절 보정값.
# M0609 관절 순서가 joint_1~joint_6이면 index 1=joint_2, index 2=joint_3이다.
# 만약 실행했을 때 박스가 내려가면 JOINT_LIFT_SIGN을 -1.0으로 바꿔라.
JOINT_LIFT_SIGN = 1.0
JOINT_LIFT_J2_DELTA_RAD = -0.61   # 108_: ㄱ자 리프트 목표. 초기 link_2 z=-90 기준에서 약 -125도까지 허용/이동
JOINT_LIFT_J3_DELTA_RAD =  0.12   # 108_: 106 성공값에 가깝게 보조. 높이가 부족하면 0.18~0.25로 소폭 증가
JOINT_USE_J3_FOR_LIFT = True      # link_2만 테스트하려면 False로 바꿔라.

# 108_: Stage에서 초기 link_2 Orient Z가 -90도로 보인다는 사용자의 기준을 joint_2 제한으로 변환한다.
# 계산식: link2_z_est_deg = initial_link2_z_deg + sign * degrees(current_j2 - reference_j2)
# 초기 상태에서 reference_j2를 자동으로 읽고, link2_z_est_deg >= min_link2_z_deg가 되도록 joint_2 target을 clamp한다.
LINK2_ORIENT_Z_GUARD_ENABLED = True
LINK2_ORIENT_Z_INITIAL_DEG = -90.0
LINK2_ORIENT_Z_MIN_DEG = -135.0
LINK2_ORIENT_Z_TARGET_DEG = -125.0  # 108_: 계산 목표값. ref_j2 + radians(-125 - -90) ≈ ref_j2 -0.611rad
LINK2_ORIENT_Z_SIGN = 1.0  # 실행 결과 GUI Orient Z 변화 방향이 반대면 -1.0으로 바꾼다.
LINK2_ORIENT_Z_GUARD_LOG = True

# 반대편으로 넘기는 동작. 반대방향이 이상하면 JOINT_SWING_SIGN만 -1.0으로 바꿔라.
JOINT_SWING_SIGN = 1.0
JOINT_SWING_DELTA_RAD = 3.10      # 약 178도. 거의 반 바퀴 회전. 너무 많이 돌면 2.85~3.00으로 낮춘다.
JOINT_SWING_CLAMP_RAD = 6.28      # 너무 과회전 방지

# 내려놓기. lift_delta를 몇 % 되돌릴지 결정한다.
# 1.0이면 들어올린 만큼 거의 전부 내리고, 0.55면 높은 위치에서 살짝 내린 뒤 release한다.
JOINT_LOWER_RETURN_RATIO = 0.0  # 75_: swing 후 j2/j3를 되돌리지 않음. link_1 z축 회전만 사용
JOINT_RELEASE_MIN_BOX_CENTER_Z = 0.0   # 75_: lower phase 비활성화. 높이는 APalt slot snap에서 맞춤
JOINT_RELEASE_MIN_SUCTION_Z = 0.0      # 75_: lower guard 비활성화

# 27_: 벽을 뚫지 않기 위한 핵심 조건. 이 높이 전에는 joint_1 회전 금지.
JOINT_STRICT_VERTICAL_LIFT_BEFORE_SWING = True
JOINT_SWING_START_MIN_BOX_CENTER_Z = 1.02  # 108_: joint_2 guard 때문에 lift 높이가 낮아질 수 있어 임시 기준 완화
JOINT_SWING_START_MIN_SUCTION_Z = 1.14  # 108_: joint_2 guard 상태에서 swing 시작 기준
JOINT_LIFT_EXTRA_HOLD_STEPS = 55

# 28_ 조정 요약:
# - 박스가 벽에 닿아 멈춘 상태를 기준으로, 흡착 직후에는 절대 joint_1을 돌리지 않는다.
# - box/suction 높이가 기준 이상 올라간 뒤에만 joint_1 반 바퀴 회전을 시작한다.
# - 내려놓기도 너무 낮게 하지 않아 벽/큐브/로봇 관통 느낌을 줄인다.
# 조정 팁:
# - 너무 많이 돌면 JOINT_SWING_DELTA_RAD = 2.85~3.00
# - 반대 방향이면 JOINT_SWING_SIGN = -1.0
# - 더 높이 들고 싶으면 J2=-0.54, J3=0.66 쪽으로 조금씩 증가
# - lift 후 오래 멈춘다면 아래 JOINT_SWING_START_MIN_* 값을 조금 낮춘다.

# 로봇을 reset 때 0 자세로 보낼지 여부.
# 처음 팔이 너무 팍 꺾이면 False로 바꿔서 현재 자세 유지부터 테스트해라.
RESET_ROBOT_TO_ZERO = True

# 30_ 핵심: 성공/실패 후 pause로 끝내지 않고 계속 반복한다.
# 성공: release -> home -> 다음 상자 대기
# 실패: attach 실패/관통 위험 -> home -> 현재 박스 위치를 다시 bbox로 읽고 재시도
RUN_CONTINUOUS_LOOP = True  # 135_: release 후 멈추지 않고 home 복귀/대기 루프 계속 진행. forklift 테스트용
LOOP_RETRY_AFTER_ATTACH_FAIL = True
LOOP_IGNORE_RELEASED_BOX_UNTIL_MOVED = False  # 47_: OriBoxA_* root 완료 목록으로 재집기 방지하므로 이동거리 ignore는 끈다.
LOOP_RELEASED_BOX_IGNORE_MOVE_TOL = 0.20  # release된 같은 박스를 바로 다시 집지 않기 위한 XY 이동 기준(m)
LOOP_WAIT_LOG_INTERVAL = 30

# ╔══════════════════════════════════════════════════════════════╗
# ║  D. 박스 정지 후 흡착 설정                                    ║
# ╚══════════════════════════════════════════════════════════════╝
# 핵심 변경: 박스가 움직이는 중에는 로봇이 pick-place를 시작하지 않는다.
# 8_ 수정: "사진에서 선택한 Small_Cardboard_box가 화면상 멈췄는지"를 기준으로 본다.
# 이전 7_는 physics:angularVelocity까지 정지 조건에 넣어서, bbox가 완전히 멈춰도
# angularVelocity 잔류값 때문에 stable=0/25에서 영원히 못 넘어갈 수 있었다.
WAIT_UNTIL_BOX_STOPPED_BEFORE_PICK = True
BOX_STABLE_REQUIRED_STEPS = 25      # 25 step 연속 정지 판정 후 시작. 너무 오래 기다리면 15~20으로 낮춘다.
BOX_STABLE_POS_TOL = 0.0015         # m/step. 박스 bbox 중심 이동량이 1.5mm 이하이면 정지 후보.
BOX_STABLE_LINEAR_VEL_TOL = 0.010   # m/s. 현재는 로그용 기본값. 아래 USE_LINEAR_VEL=True일 때만 gate에 사용.
BOX_STABLE_ANGULAR_VEL_TOL = 0.050  # rad/s. 현재는 로그용 기본값. 아래 USE_ANGULAR_VEL=True일 때만 gate에 사용.

# 정지 판정에 실제로 사용할 항목.
# 사용자 요청 기준: /World/OriBoxA/Small_Cardboard_box prim 하나가 "안 움직일 때"만 본다.
# 그래서 bbox 중심 이동량만 gate에 사용하고, linear/angular velocity는 정확한 원인 확인용 로그로만 출력한다.
BOX_STOP_USE_BBOX_MOVE = True
BOX_STOP_USE_LINEAR_VEL = False
BOX_STOP_USE_ANGULAR_VEL = False

BOX_FREEZE_AFTER_STOP = False       # 18_: 박스는 컨베이어가 굴려오게 둔다. 정지 판정 후에도 강제 kinematic 처리하지 않는다.
# 15_ 핵심: 흡착 후에는 실제 물리 충돌/중력과 scripted 이동이 싸우지 않도록
# 박스 subtree의 rigid body / collision을 잠시 끄고 순수 visual scripted carry로 이동한다.
# VGC10은 visual-only이므로 실제 물리로 박스를 밀 수 없다. 시연 안정성을 위해 이 방식이 가장 확실하다.
BOX_DISABLE_PHYSICS_DURING_CARRY = False  # 51_: 박스 물리 OFF/scripted carry 금지. FixedJoint 물리 연결로 테스트
BOX_REENABLE_PHYSICS_AFTER_RELEASE = False  # 66_: 적재/완료 박스는 다음 사이클에서 물리를 다시 켜지 않는다. root-child 순간이동 방지.
# 56_: FixedJoint로 잡으면 떨어진 body 사이 joint가 스냅/회전을 만들 수 있다.
# 그래서 실제 윗면 판정 후에는 top-lock kinematic carry로 조용히 들고 이동한다.
# 적재된 박스는 kinematic 상태를 유지해서 2x2 스택이 굴러가지 않게 한다.
BOX_KEEP_KINEMATIC_AFTER_RELEASE = False
BOX_STOP_LOG_INTERVAL = 10          # 69_: 새 환경에서는 로그 폭주 방지
BOX_STOP_LOG_EVERY_STEP = False

# 18_ 핵심: 박스가 시작 위치에서 잠깐 정지해 보인다고 바로 pick하지 않는다.
# 박스는 컨베이어가 굴려와서 실제 픽업 구간에 들어온 뒤 멈춰야 한다.
BOX_READY_ZONE_GATE = True
BOX_READY_MIN_Y = -4.00          # 이 y보다 작아진 뒤에만 pick 시작. 시작점 y=-2.8 근처 오검출 방지.
BOX_READY_MAX_CENTER_Z = 1.05    # 공중/초기 스폰 높이에서 정지 판정되는 것 방지. 실제 벨트 위 center z≈0.888.
RESET_BOX_TO_USD_START_ON_PLAY = False  # 20_: 박스는 컨베이어가 자연스럽게 굴려오게 둔다. 시작 시 코드로 위치를 되돌리지 않는다.


# ╔══════════════════════════════════════════════════════════════╗
# ║  E. 1번 로봇 앞 감지 영역 기반 Pick Trigger                  ║
# ╚══════════════════════════════════════════════════════════════╝
# 41_ 핵심:
# 기존 방식은 "박스가 완전히 멈춘 뒤" pick을 시작했다.
# 이번 방식은 1번 로봇 앞의 좁은 감지 영역 안으로 박스 중심이 들어오면 바로 pick 준비/동작을 시작한다.
# 카메라 인식이 아니라 USD bbox 기반의 영역 센서 방식이다.
# 영역은 m0609_ prefix와 겹치지 않는 이름으로 둔다.
# 그래야 idle 로봇 탐색에서 pick zone visual이 로봇 후보로 잘못 잡히지 않는다.
PICK_TRIGGER_MODE = "FRONT_ZONE"  # "FRONT_ZONE" 또는 "STOPPED_BOX"

PICK_ZONE_VISUAL_ENABLED = False  # 122_: 새 USD에 직접 만든 pick_ready_zone_A를 삭제 후 재생성하지 않음  # 81_: ready zone A visual은 그대로 생성/표시한다.
# 42_ 수정: GUI에서 사용자가 맞춘 pick-ready 영역 값 적용.
# Transform: translate=(-2.08983, -4.35, 0.88), rotate=(0,0,0), scale=(0.92,0.32,0.44).
PICK_ZONE_PATH = "/World/pick_ready_zone_A"

# 영역 위치/크기 조정은 여기만 수정하면 된다. 단위는 meter, world 좌표 기준.
# center: 영역 중심, size: X/Y/Z 방향 크기.
# 처음 위치가 안 맞으면 Isaac Sim에서 박스가 지나가는 위치를 보고 아래 값을 조금씩 조정해라.
PICK_ZONE_CENTER = np.array([-2.08983, -4.35, 0.88], dtype=float)
PICK_ZONE_SIZE   = np.array([0.92,  0.32, 0.44], dtype=float)

# 박스 bbox center가 영역 안에 몇 step 연속 들어와야 pick을 시작할지.
# 1~3이면 빠르고, 8~15면 오검출이 줄어든다.
PICK_ZONE_REQUIRED_STEPS = 3
PICK_ZONE_LOG_INTERVAL = 10
PICK_ZONE_LOG_EVERY_STEP = False

# True이면 영역 진입 순간 박스 velocity를 0으로 정리해서, 이동 중인 박스를 너무 놓치는 현상을 줄인다.
# 실제 컨베이어 흐름을 더 살리고 싶으면 False로 바꿔라.
PICK_ZONE_ZERO_BOX_VELOCITY_ON_PICK_START = True
# 77_: A ready zone 안에 들어온 뒤, 바로 집지 않고 bbox 기준으로 정지까지 확인한다.
PICK_ZONE_REQUIRE_STOPPED_BEFORE_PICK = True
# 77_: ready zone A에서는 OriBoxA 계열만 집는다. OriBoxB는 나중에 B zone을 따로 만들 때 사용한다.
PICK_ZONE_A_ONLY = True

# 감지 영역 visual. 물리 없음. 색만 들어간 표시용 큐브다.
PICK_ZONE_COLOR = np.array([0.05, 0.85, 0.20], dtype=float)  # RGB, 초록색
PICK_ZONE_OPACITY = 0.28

# 65_ 핵심: robotAprop_01 충돌 확인/보정.
# 주의: 현재 운반 방식은 VGC10이 실제로 박스를 밀고 가는 물리 운반이 아니라,
# 흡착된 박스 root를 suction 위치에 맞춰 직접 따라가게 하는 top-lock 방식이다.
# 그래서 collider가 있어도 직접 Xform 이동은 물리 충돌로 자동 정지하지 않는다.
# 아래 설정은 1) robotAprop_01에 static collider를 강제로 켜고,
# 2) 운반 경로의 Z 높이를 robotAprop_01 윗면보다 충분히 높여 뚫고 지나가는 장면을 피한다.
ROBOTAPROP_COLLISION_GUARD_ENABLED = False  # 77_: 예전 robotAprop/Cube 보정 로직 사용 안 함. /World/APalt Mesh를 받침 기준으로 사용
ROBOTAPROP_NAME_PREFIXES = ("robotAprop", "RobotAprop", "robotaprop", "Robotaprop")
ROBOTAPROP_EXACT_PATH_CANDIDATES = (
    "/World/robotAprop_01",
    "/World/RobotAprop_01",
    "/World/robotaprop_01",
    "/World/Robotaprop_01",
)
ROBOTAPROP_STATIC_COLLIDER_REPAIR = False  # 77_: USD에서 직접 편집한 robotAprop 물리 속성 건드리지 않음
ORIBOX_COLLIDER_REPAIR_ON_SETUP = False  # 77_: root RigidBody/child Collider는 USD에서 이미 정리했으므로 코드가 재보정하지 않음
ROBOTAPROP_CLEARANCE_Z_MARGIN = 0.080  # robotAprop_01 윗면과 박스 바닥 사이 여유 높이
ROBOTAPROP_CLEARANCE_EXTRA_SUCTION_Z = 0.015  # suction 목표 자체에 추가 여유
ROBOTAPROP_COLLISION_LOG_INTERVAL = 30
ADD_DEFAULT_GROUND_PLANE = False  # 77_: USD에 있는 환경만 사용. 코드가 default ground plane을 새로 만들지 않음


# ╔══════════════════════════════════════════════════════════════╗
# ║  F. OriBoxA_ 다중 박스 + BoxAprop 위 2x2 적재 설정              ║
# ╚══════════════════════════════════════════════════════════════╝
# 52_ 핵심:
# - /World 하위에서 이름이 OriBoxA_ 로 시작하는 박스만 대상으로 삼는다.
# - 각 root 하위의 Small_Cardboard_box bbox center가 위 PICK_ZONE 안에 들어오면 pick 대상으로 선택한다.
# - 1차 목표: BoxAprop 위에 2개 나란히 적재 = (2,1)
# - 2차 목표: 2개 적재 후 BoxAprop를 박스 높이만큼 낮추고 다시 2개 적재 = (2,2)
# - BoxAprop의 위치/크기는 사용자가 USD에서 직접 조정하면 되고, 코드는 매번 BoxAprop world bbox를 읽는다.
MULTI_ORIBOX_STACKING_ENABLED = True
ORIBOX_STACK_PARENT_PATH = "/World"
ORIBOX_STACK_NAME_PREFIX = "OriBoxA_"  # legacy 표시용
ORIBOX_STACK_NAME_PREFIXES = ("OriBoxA_", "OriBoxA")  # 77_: ready zone A에서는 A 상자만 대상
ORIBOX_STACK_BOX_MESH_NAME = "Small_cardboard_box"  # 72_: child 이름은 collider 확인/호환용. 좌표 기준은 root
ORIBOX_STACK_BOX_MESH_NAME_CANDIDATES = ("Small_cardboard_box", "Small_Cardboard_box")  # USD 대소문자 차이 fallback

STACK_SUPPORT_CUBE_PATH = "/World/APalt"  # 117_: 상자를 올려둘 받침 큐브/APalt Mesh 기준  # 77_: 예전 /World/Cube 받침 기준 제거. 팔레트 기준은 /World/APalt만 사용
STACK_SUPPORT_NAME_CANDIDATES = ("APalt",)  # 117_: fallback도 APalt만 허용  # 77_: fallback도 APalt만 허용

# 75_ 핵심: 팔레타이징 기준은 USD에 사용자가 찍어둔 Xform /World/APalt 를 사용한다.
# APalt는 "팔레트 윗면 중심" 좌표로 둔다. 실제 slot z는 APalt.z + box_height/2 로 계산한다.
USE_PALLET_ORI_A_MARKER = False  # 117_: /World/APalt Mesh의 bbox top을 받침 윗면으로 사용
PALLET_ORI_A_PATH = "/World/APalt"  # 117_: APalt를 팔레트/받침 기준으로 사용
PALLET_SLOT_AXIS = "ROBOT_X"  # 2x1 방향. 필요하면 "ROBOT_Y"로 바꿔 테스트
STACK_COLUMNS = 2  # 145_: APalt 위 좌/우 2열 기준. 홀수=160도, 짝수=130도
STACK_LAYERS = 2      # 145_: 최대 2층까지 테스트
STACK_SLOT_COUNT = 4  # 145_: 최대 4개 적재. 실제 이번 테스트 개수는 FORKLIFT_TRIGGER_COUNT가 결정
STACK_SLOT_AXIS = "ROBOT_X"  # 60_: 로봇 기준 좌우 방향으로 2개를 나란히 둔다.
STACK_SLOT_GAP = 0.550  # 79_: 화면상 붙어 보여서 오차 고려. box_x≈0.26 + gap 0.16 = slot 간격 약 0.42m
STACK_FIRST_SLOT_OFFSET = np.array([0.0, 0.0, 0.0], dtype=float)  # world 보정 fallback
# 124_ APalt와 상자 간격 조절 핵심 파라미터
# BOXAPROP_SLOT_OFFSET_ROBOT = [좌우, 앞뒤, 높이] 보정값(m), 로봇 기준.
#   첫 번째 값 + : 로봇 기준 오른쪽, - : 왼쪽
#   두 번째 값 + : 로봇 기준 앞쪽/바깥쪽, - : 뒤쪽/안쪽  (방향이 반대로 보이면 부호만 바꿔 테스트)
#   세 번째 값 + : 더 위에서 release, - : 더 낮게 release
BOXAPROP_SLOT_OFFSET_ROBOT = np.array([0.0, 0.0, 0.0], dtype=float)
# APalt 윗면과 상자 바닥 사이 높이 여유(m).
#   키우면 더 위에서 떨어짐/덜 파묻힘, 줄이면 APalt에 더 가까이 붙음.
STACK_PLACE_Z_CLEARANCE = 0.006
STACK_USE_BOX_SIZE_FOR_SLOT_STEP = True
STACK_MANUAL_SLOT_STEP = np.array([0.80, 0.0, 0.0], dtype=float)  # 79_: bbox 계산 fallback도 42cm 간격으로 고정

# 1층 2개를 놓은 뒤 BoxAprop를 박스 높이만큼 낮춘다.
# 그러면 1층 박스 윗면이 다시 원래 작업 높이 근처가 되고, 같은 방식으로 2층 2개를 쌓을 수 있다.
STACK_LOWER_SUPPORT_AFTER_EACH_LAYER = True   # 145_: 3개 이상 테스트 시 1층 2개 후 APalt를 박스 높이만큼 내림
STACK_LOWER_SUPPORT_STEPS = 180                 # 145_: APalt lowering step 수. 크면 더 천천히 내려감
STACK_LOWER_EXTRA_Z = 0.000                     # 양수면 박스 높이보다 조금 더 낮춤
STACK_LOWER_MAX_LAYERS = 1                      # 145_: 최대 1번만 내림(1층 -> 2층 준비)

# 성공한 root는 다시 집지 않는다. Stop→Play하면 이 목록은 코드 내부에서 초기화된다.
STACK_SKIP_COMPLETED_BOXES = True
STACK_STOP_WHEN_FULL = True  # 75_: 2x1 두 칸이 차면 정지
# 52_: 이번 버전은 박스 물리 OFF/scripted carry가 아니라 FixedJoint + 관절 기반 이동 우선.
STACK_USE_DIRECT_CARRY_ROUTE = True
# 순간이동 보정 금지. BoxAprop 위 slot 중심 강제 snap을 쓰지 않는다.
STACK_SNAP_BOX_TO_SLOT_ON_RELEASE = False  # 77_: 순간이동 금지. APalt 슬롯으로 강제 보정하지 않고 현재 위치에서 release
STACK_SNAP_MAX_XY_ERROR = 0.0  # 75_: snap 비활성화

# 64_ 핵심 디버그: 박스가 순간이동하는 원인을 잡기 위해 지정 박스의 world pose를 계속 추적한다.
# 기본은 네가 요청한 OriBoxA_02 / OriBoxA_03만 추적한다.
# 로그가 너무 많으면 POSE_TRACK_LOG_EVERY_STEP=False, POSE_TRACK_LOG_INTERVAL=10 정도로 바꿔라.
POSE_TRACK_ENABLED = False  # 128_: 일반 POSE_TRACK 로그 제거. release 진단 로그만 출력
POSE_TRACK_ROOT_PATHS = [
    "/World/OriBoxA_01",
    "/World/OriBoxA_02",
]
POSE_TRACK_BOX_CHILD_NAME = "Small_cardboard_box"  # 72_: log에서 child 동반 여부 확인용. 기준 좌표/bbox는 root
POSE_TRACK_LOG_EVERY_STEP = False  # 69_: 새 환경에서는 로그 폭주 방지. 정밀 추적 필요하면 True
POSE_TRACK_LOG_INTERVAL = 10
POSE_TRACK_JUMP_WARN_TOL = 0.015  # m. 이전 로그 대비 1.5cm 이상 바뀌면 JUMP 표시
_POSE_TRACK_STATE = {"step": 0, "prev": {}}

# 72_ 핵심:
# 새 Conveyor_lift.usd에서는 USD에서 root에 Rigid Body를 붙이고 child는 Collider만 남긴 구조를 사용한다.
# root가 실제 물리 박스의 대표 좌표이므로, 코드에서 child를 root에 강제로 맞추지 않는다.
ORIBOX_KEEP_CHILD_CENTER_MATCHED_TO_ROOT = False
ORIBOX_CENTER_MATCH_TOL = 0.001
ORIBOX_CENTER_MATCH_LOG = False


# ============================================================
# 유틸
# ============================================================
def find_prim_path_by_name(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def resolve_stack_support_path(stage):
    """
    69_ FIX:
    새 Conveyor_lift 작업환경은 기존 BoxAprop 대신 /World/Cube 같은 받침 prim을 쓸 수 있다.
    우선 STACK_SUPPORT_CUBE_PATH를 확인하고, 없으면 후보 이름(Cube/BoxAprop/robotAprop_01)을 Stage 전체에서 찾는다.
    """
    preferred = str(STACK_SUPPORT_CUBE_PATH)
    prim = stage.GetPrimAtPath(preferred)
    if prim and prim.IsValid():
        return preferred

    world_prim = stage.GetPrimAtPath("/World")
    names = tuple(str(x) for x in globals().get("STACK_SUPPORT_NAME_CANDIDATES", ("Cube", "BoxAprop")))
    if world_prim and world_prim.IsValid():
        for p in Usd.PrimRange(world_prim):
            try:
                if p.GetName() in names:
                    return str(p.GetPath())
            except Exception:
                pass

    # 마지막 fallback: 기존 경로를 반환해서 이후 bbox None 경고가 제대로 뜨게 한다.
    return preferred


def _safe_normalize_vec(v, fallback):
    v = np.array(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array(fallback, dtype=float)
    return v / n


def get_robot_frame_axes_for_stack(stage):
    """
    BoxAprop 슬롯을 '로봇 기준'으로 계산하기 위한 좌표축.

    반환:
    - origin: active robot root의 world translation
    - right : robot local +X 방향
    - forward: robot local +Y 방향
    - up    : robot local +Z 방향

    로봇 transform을 못 읽으면 world 축을 fallback으로 사용한다.
    """
    origin = get_world_translation(stage, ACTIVE_ROBOT_ROOT_PATH)
    if origin is None:
        origin = get_world_translation(stage, ROBOT_PRIM_PATH)
    if origin is None:
        origin = np.array([0.0, 0.0, 0.0], dtype=float)

    mat = get_world_matrix(stage, ACTIVE_ROBOT_ROOT_PATH)
    if mat is None:
        mat = get_world_matrix(stage, ROBOT_PRIM_PATH)

    right = np.array([1.0, 0.0, 0.0], dtype=float)
    forward = np.array([0.0, 1.0, 0.0], dtype=float)
    up = np.array([0.0, 0.0, 1.0], dtype=float)

    if mat is not None:
        try:
            rx = mat.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
            ry = mat.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
            rz = mat.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
            right = np.array([float(rx[0]), float(rx[1]), float(rx[2])], dtype=float)
            forward = np.array([float(ry[0]), float(ry[1]), float(ry[2])], dtype=float)
            up = np.array([float(rz[0]), float(rz[1]), float(rz[2])], dtype=float)
        except Exception:
            # Gf.Matrix 환경 차이 fallback
            try:
                right = np.array([float(mat[0][0]), float(mat[0][1]), float(mat[0][2])], dtype=float)
                forward = np.array([float(mat[1][0]), float(mat[1][1]), float(mat[1][2])], dtype=float)
                up = np.array([float(mat[2][0]), float(mat[2][1]), float(mat[2][2])], dtype=float)
            except Exception:
                pass

    right = _safe_normalize_vec(right, [1.0, 0.0, 0.0])
    forward = _safe_normalize_vec(forward, [0.0, 1.0, 0.0])
    up = _safe_normalize_vec(up, [0.0, 0.0, 1.0])
    return np.array(origin, dtype=float), right, forward, up


def robot_relative_vector(stage, world_pos):
    """
    world 좌표를 active robot 기준 [right, forward, up] 성분으로 변환해 로그에 찍기 위한 함수.
    사용자가 Translate 좌표를 다시 줄 때, 이 값을 보고 어느 방향으로 보정할지 판단한다.
    """
    origin, right, forward, up = get_robot_frame_axes_for_stack(stage)
    v = np.array(world_pos, dtype=float) - np.array(origin, dtype=float)
    return np.array([
        float(np.dot(v, right)),
        float(np.dot(v, forward)),
        float(np.dot(v, up)),
    ], dtype=float)


def get_task_completed_root_path(task):
    """
    66_: 실제 이동은 OriBox root를 직접 움직이며,
    완료/재집기 방지 기준은 /World/OriBoxA_02 같은 root path로 유지한다.
    """
    root = getattr(task, "active_box_root_path", None)
    if root:
        return str(root)
    move = getattr(task, "box_move_path", None)
    return str(move) if move else ""


def find_robotaprop_paths(stage):
    """Stage 안에서 robotAprop_01 계열 prim을 찾는다."""
    found = []

    for path in ROBOTAPROP_EXACT_PATH_CANDIDATES:
        try:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                found.append(str(prim.GetPath()))
        except Exception:
            pass

    world = stage.GetPrimAtPath("/World")
    if world and world.IsValid():
        prefixes = tuple(str(x).lower() for x in ROBOTAPROP_NAME_PREFIXES)
        for prim in Usd.PrimRange(world):
            try:
                name_l = prim.GetName().lower()
                if any(name_l.startswith(p) for p in prefixes):
                    found.append(str(prim.GetPath()))
            except Exception:
                pass

    # 하위 prim이 같이 잡히면 상위 경로만 남긴다.
    unique = []
    for p in sorted(set(found), key=len):
        if not any(p.startswith(q + "/") for q in unique):
            unique.append(p)
    return unique


def _prim_is_geometry_like(prim):
    try:
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Sphere):
            return True
    except Exception:
        pass
    return False


def enable_collision_on_subtree(stage, root_path, static=True, label="COLLIDER"):
    """
    root 하위 geometry에 CollisionAPI를 켠다.
    static=True이면 RigidBodyAPI/MassAPI를 제거해서 고정 장애물 collider처럼 둔다.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return {"root": root_path, "valid": False, "collision": 0, "rigid_removed": 0, "mesh_approx": 0}

    collision_count = 0
    rigid_removed = 0
    mesh_approx = 0

    for prim in Usd.PrimRange(root):
        try:
            is_geom = _prim_is_geometry_like(prim)
            has_col = prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAttribute("physics:collisionEnabled")
            if is_geom or has_col:
                col_api = UsdPhysics.CollisionAPI.Apply(prim)
                attr = prim.GetAttribute("physics:collisionEnabled")
                if not attr:
                    attr = col_api.CreateCollisionEnabledAttr(True)
                attr.Set(True)
                collision_count += 1
        except Exception:
            pass

        try:
            if _prim_is_geometry_like(prim):
                mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
                approx_attr = prim.GetAttribute("physics:approximation")
                if not approx_attr:
                    approx_attr = mesh_col.CreateApproximationAttr()
                # Box/convexHull가 지원되는 환경이 다르므로 실패하면 그냥 CollisionAPI만 유지한다.
                try:
                    approx_attr.Set("convexHull")
                    mesh_approx += 1
                except Exception:
                    pass
        except Exception:
            pass

        if static:
            try:
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    rigid_removed += 1
            except Exception:
                pass
            try:
                if prim.HasAPI(UsdPhysics.MassAPI):
                    prim.RemoveAPI(UsdPhysics.MassAPI)
            except Exception:
                pass
            try:
                rb_attr = prim.GetAttribute("physics:rigidBodyEnabled")
                if rb_attr:
                    rb_attr.Set(False)
            except Exception:
                pass
            try:
                kin_attr = prim.GetAttribute("physics:kinematicEnabled")
                if kin_attr:
                    kin_attr.Set(False)
            except Exception:
                pass

    return {"root": root_path, "valid": True, "collision": collision_count, "rigid_removed": rigid_removed, "mesh_approx": mesh_approx}


def repair_robotaprop_and_oribox_colliders(stage, completed_roots=None, verbose=True):
    """robotAprop_01은 static collider, OriBoxA_*는 collider를 명시적으로 켠다."""
    if completed_roots is None:
        completed_roots = set()

    if ROBOTAPROP_STATIC_COLLIDER_REPAIR:
        paths = find_robotaprop_paths(stage)
        if verbose:
            print(f"  [ROBOTAPROP_SEARCH] found={paths if paths else 'NONE'}")
        for p in paths:
            stats = enable_collision_on_subtree(stage, p, static=True, label="ROBOTAPROP")
            if verbose:
                print(
                    f"  [ROBOTAPROP_COLLIDER] root={p}, collision_on={stats.get('collision')}, "
                    f"rigid_removed={stats.get('rigid_removed')}, mesh_approx={stats.get('mesh_approx')}"
                )

    if ORIBOX_COLLIDER_REPAIR_ON_SETUP:
        candidates = discover_oriboxa_stack_candidates(stage, completed_roots=set(completed_roots)) if 'discover_oriboxa_stack_candidates' in globals() else []
        count = 0
        for c in candidates:
            root_path = c.get("root_path")
            if not root_path or root_path in completed_roots:
                continue
            stats = enable_collision_on_subtree(stage, root_path, static=False, label="ORIBOX")
            count += int(stats.get("collision", 0) or 0)
        if verbose:
            print(f"  [ORIBOX_COLLIDER_REPAIR] candidates={len(candidates)}, collision_on_total={count}")


def get_robotaprop_clearance_suction_z(stage, box_height, attached_center_offset):
    """
    direct Xform/top-lock 운반은 실제 충돌로 멈추지 않으므로,
    robotAprop_01 위를 지나갈 때 박스 바닥이 robotAprop_01 top보다 위에 있도록 suction z를 올린다.
    """
    if not ROBOTAPROP_COLLISION_GUARD_ENABLED:
        return None, "disabled"

    paths = find_robotaprop_paths(stage)
    if not paths:
        return None, "robotAprop_not_found"

    max_top_z = None
    used = []
    for p in paths:
        info = get_world_bbox_info(stage, p)
        if info is None:
            continue
        top_z = float(info["max"][2])
        used.append((p, top_z))
        if max_top_z is None or top_z > max_top_z:
            max_top_z = top_z

    if max_top_z is None:
        return None, f"robotAprop_bbox_none paths={paths}"

    # box_center = suction + attached_center_offset.
    # box_bottom = box_center_z - box_height/2.
    # box_bottom >= robotAprop_top + margin 이 되도록 suction_z를 계산한다.
    offset_z = float(np.array(attached_center_offset, dtype=float)[2])
    required = (
        float(max_top_z)
        + float(box_height) * 0.5
        + float(ROBOTAPROP_CLEARANCE_Z_MARGIN)
        - offset_z
        + float(ROBOTAPROP_CLEARANCE_EXTRA_SUCTION_Z)
    )
    return float(required), f"max_top_z={max_top_z:.4f}, box_h={box_height:.4f}, offset_z={offset_z:.4f}, used={used}"


def initialize_robot(robot, world):
    robot.initialize()
    # RG2를 삭제했기 때문에 robot.gripper는 사용하지 않는다.
    robot.set_joint_positions(np.zeros(robot.num_dof))


def clear_prim_if_exists(stage, prim_path: str):
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)


def remove_goal_marker(stage, verbose=True):
    """
    /World/goal_marker는 목표 위치 확인용 visual 큐브일 뿐이라 시연에서는 제거한다.
    Conveyor.usd reference 안에 이미 들어있어 RemovePrim만으로 남는 경우를 대비해
    invisible + active=False까지 같이 처리한다.
    """
    prim = stage.GetPrimAtPath(GOAL_MARKER_PATH)
    if not prim.IsValid():
        return False

    try:
        stage.RemovePrim(GOAL_MARKER_PATH)
    except Exception:
        pass

    # reference로 compose된 prim은 RemovePrim 뒤에도 보일 수 있으므로 fallback 처리
    prim = stage.GetPrimAtPath(GOAL_MARKER_PATH)
    if prim.IsValid():
        try:
            UsdGeom.Imageable(prim).MakeInvisible()
        except Exception:
            pass
        try:
            prim.SetActive(False)
        except Exception:
            pass

    if verbose:
        still_valid = stage.GetPrimAtPath(GOAL_MARKER_PATH).IsValid()
        print(f"  [OK] goal_marker 제거/비활성화: {GOAL_MARKER_PATH}, valid_after={still_valid}")
    return True


def disable_legacy_camera_graphs(stage, verbose=True):
    """
    106_: 카메라 복구 반영.
    예전 코드에서는 오류 회피를 위해 /World/rsd455까지 SetActive(False) 했지만,
    지금은 rsd455를 사용해야 하므로 /World/rsd455는 끄지 않는다.

    - robot 하위 오래된 camera_graph만 필요 시 비활성화
    - /World/rsd455는 active=True로 유지
    """
    targets = []
    if bool(globals().get("DISABLE_LEGACY_ROBOT_CAMERA_GRAPHS", True)):
        targets.extend([
            "/World/m0609_A/Graph/camera_graph",
            "/World/m0609_B/Graph/camera_graph",
        ])
    if bool(globals().get("DISABLE_WORLD_RSD455_PRIM", False)):
        targets.append(str(globals().get("RSD455_ROOT_PATH", "/World/rsd455")))

    changed = []
    for path in targets:
        try:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                prim.SetActive(False)
                changed.append(path)
        except Exception:
            pass

    if bool(globals().get("KEEP_RSD455_CAMERA_ACTIVE", True)):
        try:
            rsd_path = str(globals().get("RSD455_ROOT_PATH", "/World/rsd455"))
            rsd = stage.GetPrimAtPath(rsd_path)
            if rsd and rsd.IsValid():
                rsd.SetActive(True)
                try:
                    UsdGeom.Imageable(rsd).MakeVisible()
                except Exception:
                    pass
                if verbose:
                    print(f"  [CAMERA_RESTORE_106] keep active: {rsd_path}, type={rsd.GetTypeName()}, active={rsd.IsActive()}")
                # 하위 Camera prim이 실제로 있는지 확인한다.
                camera_children = []
                for p in Usd.PrimRange(rsd):
                    try:
                        if p.GetTypeName() == "Camera" or p.IsA(UsdGeom.Camera):
                            camera_children.append(str(p.GetPath()))
                    except Exception:
                        pass
                if verbose:
                    print(f"  [CAMERA_RESTORE_106] camera children={camera_children if camera_children else 'NONE_FOUND_UNDER_RSD455'}")
            elif verbose:
                print(f"  [CAMERA_RESTORE_106][WARN] rsd455 prim not found: {rsd_path}")
        except Exception as e:
            if verbose:
                print(f"  [CAMERA_RESTORE_106][WARN] failed to restore rsd455 active: {e}")

    if verbose:
        print(f"  [CLEAN] legacy camera/ROS graph disabled={changed if changed else 'NONE'}")
    return changed




def force_show_cube_prims_109(stage, verbose=True):
    """
    109_: USD 안에 있는 Cube가 실행 중 invisible/active=false 상태로 남는 경우를 막는다.
    사용자가 사진으로 보여준 Prim Path는 /Cube이고, 기존 코드 후보는 /World/Cube라서 둘 다 처리한다.
    원본 USD를 저장하지 않고 실행 중 Stage에서만 visible/active를 보장한다.
    """
    if not bool(globals().get("FORCE_SHOW_CUBE_PRIMS_109", True)):
        return
    paths = tuple(globals().get("CUBE_VISIBLE_CANDIDATE_PATHS_109", ("/Cube", "/World/Cube")))
    shown = []
    missing = []
    for root_path in paths:
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            missing.append(root_path)
            continue
        try:
            root.SetActive(True)
        except Exception:
            pass
        for prim in Usd.PrimRange(root):
            try:
                prim.SetActive(True)
            except Exception:
                pass
            try:
                if prim.IsA(UsdGeom.Imageable):
                    img = UsdGeom.Imageable(prim)
                    img.MakeVisible()
                    # 목적이 guide/proxy로 되어 있어 viewport에서 빠질 수 있으므로 default로 보정 시도
                    try:
                        img.CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
                    except Exception:
                        pass
            except Exception:
                pass
        shown.append(root_path)
    if verbose:
        print(f"  [CUBE_VISIBLE_109] shown={shown if shown else 'NONE'}, missing={missing if missing else 'NONE'}")


def _iter_subtree_all_children_146(root):
    """Usd.PrimRange가 inactive prim을 건너뛰는 경우까지 대비해 GetAllChildren으로 하위 prim을 돈다."""
    if not root or not root.IsValid():
        return
    stack = [root]
    while stack:
        prim = stack.pop(0)
        yield prim
        try:
            children = list(prim.GetAllChildren())
        except Exception:
            try:
                children = list(prim.GetChildren())
            except Exception:
                children = []
        stack[0:0] = children


def _make_prim_visible_active_146(prim):
    if not prim or not prim.IsValid():
        return False
    changed = False
    try:
        prim.SetActive(True)
        changed = True
    except Exception:
        pass
    try:
        if prim.IsA(UsdGeom.Imageable):
            img = UsdGeom.Imageable(prim)
            img.MakeVisible()
            changed = True
            if bool(globals().get("ENVIRONMENT_FORCE_PURPOSE_DEFAULT_146", True)):
                try:
                    img.CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
                except Exception:
                    pass
    except Exception:
        pass
    return changed


def _make_ancestors_visible_active_146(stage, path_str):
    """상위 prim이 invisible이면 자식만 visible이어도 안 보이므로 상위까지 같이 복구한다."""
    try:
        p = Sdf.Path(path_str)
    except Exception:
        return
    cur = p.GetParentPath()
    while cur and str(cur) not in ("", "/"):
        prim = stage.GetPrimAtPath(cur)
        if prim and prim.IsValid():
            _make_prim_visible_active_146(prim)
        cur = cur.GetParentPath()


def force_show_environment_prims_146(stage, verbose=True):
    """
    146_: 새 USD에서 Prim Path가 Environment인 환경 오브젝트가 145 실행 후 안 보이는 문제 방지.
    - 명시 후보: /Environment, /World/Environment 등
    - 추가 후보: /World 하위에서 이름에 Environment가 들어가는 prim
    - root와 모든 하위 prim을 active=True, visible, purpose=default로 보정
    원본 USD를 저장하지 않고 실행 중 Stage에서만 처리한다.
    """
    if not bool(globals().get("FORCE_SHOW_ENVIRONMENT_PRIMS_146", True)):
        return []

    candidate_paths = []
    for p in tuple(globals().get("ENVIRONMENT_VISIBLE_CANDIDATE_PATHS_146", ())):
        if p and p not in candidate_paths:
            candidate_paths.append(str(p))

    keywords = tuple(globals().get("ENVIRONMENT_VISIBLE_NAME_KEYWORDS_146", ("Environment",)))
    scan_roots = ["/World", "/"]
    for scan_root in scan_roots:
        try:
            root = stage.GetPrimAtPath(scan_root) if scan_root != "/" else stage.GetPseudoRoot()
            if not root or not root.IsValid():
                continue
            for prim in _iter_subtree_all_children_146(root):
                try:
                    name = prim.GetName()
                    path = str(prim.GetPath())
                    if any(k in name for k in keywords) and path not in candidate_paths:
                        candidate_paths.append(path)
                except Exception:
                    pass
        except Exception:
            pass

    shown = []
    missing = []
    fixed_count = 0
    for root_path in candidate_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            missing.append(root_path)
            continue
        _make_ancestors_visible_active_146(stage, root_path)
        local_count = 0
        for prim in _iter_subtree_all_children_146(root):
            if _make_prim_visible_active_146(prim):
                local_count += 1
        fixed_count += local_count
        shown.append(f"{root_path}({local_count})")

    if verbose:
        print(f"  [ENV_VISIBLE_146] shown={shown if shown else 'NONE'}, fixed_prims={fixed_count}, missing={missing if missing else 'NONE'}")
    return shown

def _disable_collision_and_visibility(stage, root_path: str):
    """
    지정한 prim 하위 전체를 보이지 않게 하고 물리 충돌을 끈다.
    RemovePrim이 reference prim에 바로 먹지 않는 경우를 대비한 fallback이다.
    """
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return {"hidden": 0, "collision_off": 0, "rigid_removed": 0, "mass_removed": 0}

    hidden = 0
    collision_off = 0
    rigid_removed = 0
    mass_removed = 0

    for prim in list(Usd.PrimRange(root_prim)):
        try:
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
        except Exception:
            pass

        try:
            col_api = UsdPhysics.CollisionAPI.Apply(prim)
            col_api.CreateCollisionEnabledAttr(False)
            collision_off += 1
        except Exception:
            pass

        try:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                rigid_removed += 1
        except Exception:
            pass

        try:
            if prim.HasAPI(UsdPhysics.MassAPI):
                prim.RemoveAPI(UsdPhysics.MassAPI)
                mass_removed += 1
        except Exception:
            pass

    return {
        "hidden": hidden,
        "collision_off": collision_off,
        "rigid_removed": rigid_removed,
        "mass_removed": mass_removed,
    }


def _find_old_rg2_paths(stage):
    """
    onrobot_rg2ft가 예상 경로가 아닌 곳에 남아 있어도 찾아서 처리한다.
    """
    paths = set()
    if stage.GetPrimAtPath(OLD_GRIPPER_PRIM_PATH).IsValid():
        paths.add(OLD_GRIPPER_PRIM_PATH)

    robot_root = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if robot_root.IsValid():
        for prim in Usd.PrimRange(robot_root):
            name_l = prim.GetName().lower()
            path_s = str(prim.GetPath())
            if name_l == "onrobot_rg2ft" or "onrobot_rg2ft" in path_s.lower():
                paths.add(path_s)

    # 하위 prim이 먼저 잡혔을 때 상위 onrobot_rg2ft만 남긴다.
    cleaned = set()
    for p in paths:
        if "/onrobot_rg2ft" in p:
            cleaned.add(p.split("/onrobot_rg2ft")[0] + "/onrobot_rg2ft")
        else:
            cleaned.add(p)
    return sorted(cleaned, key=len, reverse=True)


def remove_old_gripper(stage, verbose=True):
    """
    기존 RG2 집게를 완전히 제거한다.

    주의: 원본 USD 파일을 지우는 게 아니라, 실행 중 Stage에서만 제거/비활성화한다.
    reference prim은 RemovePrim만으로 화면에 남는 경우가 있어서
    RemovePrim → invisible/collision off → SetActive(False) 순서로 강하게 처리한다.
    """
    paths = _find_old_rg2_paths(stage)
    if not paths:
        if verbose:
            print(f"  [INFO] 기존 RG2 집게 없음: {OLD_GRIPPER_PRIM_PATH}")
        return

    removed = 0
    deactivated = 0
    fallback_hidden = 0

    # 선택 표시가 남지 않도록 먼저 선택 해제
    try:
        omni.usd.get_context().get_selection().clear_selected_prim_paths()
    except Exception:
        pass

    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue

        # 1차: RemovePrim 시도
        try:
            stage.RemovePrim(path)
            for _ in range(5):
                simulation_app.update()
        except Exception:
            pass

        # RemovePrim 후 사라졌으면 성공
        if not stage.GetPrimAtPath(path).IsValid():
            removed += 1
            if verbose:
                print(f"  [OK] RG2 RemovePrim 성공: {path}")
            continue

        # 2차: 보이지 않게 하고 충돌 제거
        stats = _disable_collision_and_visibility(stage, path)
        fallback_hidden += stats.get("hidden", 0)

        # 3차: prim 자체 비활성화
        try:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                prim.SetActive(False)
                deactivated += 1
                for _ in range(5):
                    simulation_app.update()
        except Exception:
            pass

        if verbose:
            still_valid = stage.GetPrimAtPath(path).IsValid()
            print(f"  [OK] RG2 fallback 처리: {path} valid_after={still_valid}")
            print(f"       hidden={stats.get('hidden', 0)} collision_off={stats.get('collision_off', 0)}")

    if verbose:
        print(f"  [SUMMARY] RG2 removed={removed}, deactivated={deactivated}, hidden_children={fallback_hidden}")


def make_subtree_visual_only(stage, root_path: str):
    """
    VGC10 CAD/USD 안에 RigidBody/Collider가 들어 있으면 link_6에 붙는 순간
    바닥, 큐브, 로봇과 충돌해서 물체를 밀거나 바닥을 들어 올리는 것처럼 보일 수 있다.
    시연에서는 VGC10은 외형만 필요하므로 해당 subtree의 물리 API/충돌을 꺼 둔다.
    """
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        print(f"  [WARN] visual-only 처리 실패: prim 없음 {root_path}")
        return

    disabled_collision = 0
    removed_rigid = 0
    removed_mass = 0
    hidden_scene_extras = 0

    # 참조 USD가 compose될 시간을 조금 준다.
    for _ in range(10):
        simulation_app.update()

    for prim in list(Usd.PrimRange(root_prim)):
        name_l = prim.GetName().lower()

        # CAD 변환 중 같이 딸려온 ground/floor 같은 장면 부속물이 있으면 숨긴다.
        # VGC10 본체 이름에 ground/floor가 들어가는 경우는 거의 없고, 있더라도 visual-only 용도라 문제 적음.
        if name_l in {"ground", "groundplane", "floor", "defaultgroundplane"} or "groundplane" in name_l:
            try:
                imageable = UsdGeom.Imageable(prim)
                imageable.MakeInvisible()
                hidden_scene_extras += 1
            except Exception:
                pass

        # Collider 비활성화. HasAPI가 false여도 Apply해서 명시적으로 꺼 둔다.
        try:
            col_api = UsdPhysics.CollisionAPI.Apply(prim)
            col_api.CreateCollisionEnabledAttr(False)
            disabled_collision += 1
        except Exception:
            pass

        # RigidBody/Mass API 제거. 실패해도 무시한다.
        try:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                removed_rigid += 1
        except Exception:
            pass

        try:
            if prim.HasAPI(UsdPhysics.MassAPI):
                prim.RemoveAPI(UsdPhysics.MassAPI)
                removed_mass += 1
        except Exception:
            pass

    print("  [OK] VGC10 visual-only 처리")
    print(f"       collision disabled attrs = {disabled_collision}")
    print(f"       rigid body APIs removed  = {removed_rigid}")
    print(f"       mass APIs removed        = {removed_mass}")
    print(f"       hidden scene extras      = {hidden_scene_extras}")


def _resolve_vgc10_asset_path():
    candidates = [VGC10_USD_PATH] + list(VGC10_FALLBACK_PATHS)
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if Path(p).exists():
            return p
    return VGC10_USD_PATH


def _set_xform_common(prim, translate=None, rotate=None, scale=None):
    xform_api = UsdGeom.XformCommonAPI(prim)
    if translate is not None:
        xform_api.SetTranslate(tuple(float(x) for x in translate))
    if rotate is not None:
        xform_api.SetRotate(tuple(float(x) for x in rotate))
    if scale is not None:
        xform_api.SetScale(tuple(float(x) for x in scale))


def _get_or_create_transform_op(prim):
    """prim에 matrix transform op 하나를 만들고 계속 재사용한다."""
    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    if len(ops) == 1 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform:
        return ops[0]
    xformable.ClearXformOpOrder()
    return xformable.AddTransformOp()


def set_prim_local_matrix(prim, mat):
    op = _get_or_create_transform_op(prim)
    op.Set(mat)


def make_trs_matrix(translate=None, rotate_xyz_deg=None, scale=None):
    """T * Rz * Ry * Rx * S 순서의 local transform matrix 생성."""
    translate = np.array([0.0, 0.0, 0.0] if translate is None else translate, dtype=float)
    rotate_xyz_deg = np.array([0.0, 0.0, 0.0] if rotate_xyz_deg is None else rotate_xyz_deg, dtype=float)
    scale = np.array([1.0, 1.0, 1.0] if scale is None else scale, dtype=float)

    m = Gf.Matrix4d(1.0)
    m.SetTranslateOnly(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))

    rx = Gf.Matrix4d(1.0)
    ry = Gf.Matrix4d(1.0)
    rz = Gf.Matrix4d(1.0)
    rx.SetRotateOnly(Gf.Rotation(Gf.Vec3d(1, 0, 0), float(rotate_xyz_deg[0])))
    ry.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 1, 0), float(rotate_xyz_deg[1])))
    rz.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(rotate_xyz_deg[2])))

    sm = Gf.Matrix4d(1.0)
    sm.SetScale(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))
    return m * rz * ry * rx * sm


def get_world_matrix(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    try:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        return cache.GetLocalToWorldTransform(prim)
    except Exception:
        return None


def get_world_translation(stage, prim_path: str):
    mat = get_world_matrix(stage, prim_path)
    if mat is None:
        return None
    t = mat.ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)


def update_vgc10_visual_follow_pose(stage, robot=None):
    """
    15_ 핵심 수정:
    - VGC10 root는 /World 아래에서 tool0 월드 transform을 따라간다.
    - 실제 gripper.usda reference는 root 바로 밑이 아니라 vgc10_scaled_mount 아래에 둔다.
    - scale은 mount prim에만 한 번 적용한다.
    - 매 프레임 root pose만 갱신하므로 scale이 set_world_pose/update에 의해 사라지지 않는다.
    """
    root_prim = stage.GetPrimAtPath(VGC10_PRIM_PATH)
    if not root_prim.IsValid():
        return None

    target_mat = get_world_matrix(stage, VGC10_FOLLOW_TARGET_PATH)
    if target_mat is None and robot is not None:
        try:
            ee_pos, _ = robot.end_effector.get_world_pose()
            target_mat = Gf.Matrix4d(1.0)
            target_mat.SetTranslateOnly(Gf.Vec3d(float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])))
        except Exception:
            target_mat = None

    if target_mat is None:
        return None

    set_prim_local_matrix(root_prim, target_mat)
    t = target_mat.ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)


def attach_vgc10_to_link6(stage):
    # 중요: VGC10 CAD는 articulation 하위에 직접 붙이지 않는다.
    # /World/vgc10_visual_follow(root)가 tool0를 따라가고, 그 아래 mount에 scale을 적용한다.
    clear_prim_if_exists(stage, VGC10_PRIM_PATH)

    vgc10_asset = _resolve_vgc10_asset_path()

    root_xform = UsdGeom.Xform.Define(stage, VGC10_PRIM_PATH)
    root_prim = root_xform.GetPrim()

    mount_xform = UsdGeom.Xform.Define(stage, VGC10_MOUNT_PATH)
    mount_prim = mount_xform.GetPrim()

    model_xform = UsdGeom.Xform.Define(stage, VGC10_MODEL_PATH)
    model_prim = model_xform.GetPrim()

    # 실제 흡착 기준점. 화면에는 보이지 않는 Xform이다.
    # VGC10 root 하위에 두므로 root가 tool0를 따라갈 때 같이 따라간다.
    suction_point_xform = UsdGeom.Xform.Define(stage, VGC10_SUCTION_POINT_PATH)
    suction_point_prim = suction_point_xform.GetPrim()

    if Path(vgc10_asset).exists():
        model_prim.GetReferences().AddReference(str(vgc10_asset))
        print(f"  [OK] 실제 VGC10 USDA/USD reference 부착: {vgc10_asset}")
    else:
        print(f"  [WARN] 실제 VGC10 파일을 찾지 못함: {vgc10_asset}")
        print("         임시 visual suction pad만 생성한다. assets/gripper_vgc10_v1.usda 또는 assets/gripper.usda 경로를 확인해라.")
        UsdGeom.Cylinder.Define(stage, f"{VGC10_MODEL_PATH}/fallback_suction_pad")

    # root는 follow용이라 단위 transform으로 시작한다.
    set_prim_local_matrix(root_prim, Gf.Matrix4d(1.0))

    # scale/rotate/translate는 mount에만 적용한다. 이 방식이면 reference 내부 xform이나 set_world_pose가 scale을 덮어쓰지 못한다.
    mount_mat = make_trs_matrix(
        translate=VGC10_LOCAL_TRANSLATE,
        rotate_xyz_deg=VGC10_LOCAL_ROTATE_XYZ,
        scale=VGC10_LOCAL_SCALE,
    )
    set_prim_local_matrix(mount_prim, mount_mat)

    # model 자체는 transform 없이 reference만 둔다.
    set_prim_local_matrix(model_prim, Gf.Matrix4d(1.0))

    # suction point는 VGC10 root 기준 local offset으로 둔다.
    # 이 값은 meter 단위이며, 안 붙으면 VGC10_SUCTION_LOCAL_OFFSET만 조정한다.
    suction_mat = Gf.Matrix4d(1.0)
    suction_mat.SetTranslateOnly(Gf.Vec3d(
        float(VGC10_SUCTION_LOCAL_OFFSET[0]),
        float(VGC10_SUCTION_LOCAL_OFFSET[1]),
        float(VGC10_SUCTION_LOCAL_OFFSET[2]),
    ))
    set_prim_local_matrix(suction_point_prim, suction_mat)

    # 51_: 3x3 흡착점 그리드를 VGC10 root 하위에 생성한다.
    # p11이 중심점이고, p00~p22가 박스 윗면 접촉 판정에 사용된다.
    if SUCTION_GRID_ENABLED:
        try:
            clear_prim_if_exists(stage, SUCTION_GRID_ROOT_PATH)
            grid_root = UsdGeom.Xform.Define(stage, SUCTION_GRID_ROOT_PATH).GetPrim()
            set_prim_local_matrix(grid_root, Gf.Matrix4d(1.0))
            spacing = float(SUCTION_GRID_SPACING_XY)
            labels = []
            for iy, y_mul in enumerate([-1.0, 0.0, 1.0]):
                for ix, x_mul in enumerate([-1.0, 0.0, 1.0]):
                    label = f"p{iy}{ix}"
                    path = f"{SUCTION_GRID_ROOT_PATH}/{label}"
                    if SUCTION_GRID_MARKERS_VISIBLE:
                        point_prim = UsdGeom.Sphere.Define(stage, path).GetPrim()
                        try:
                            UsdGeom.Sphere(point_prim).CreateRadiusAttr(float(SUCTION_GRID_MARKER_RADIUS))
                            UsdGeom.Gprim(point_prim).CreateDisplayColorAttr([Gf.Vec3f(0.0, 1.0, 0.0)])
                        except Exception:
                            pass
                    else:
                        point_prim = UsdGeom.Xform.Define(stage, path).GetPrim()
                    point_mat = Gf.Matrix4d(1.0)
                    point_mat.SetTranslateOnly(Gf.Vec3d(
                        float(VGC10_SUCTION_LOCAL_OFFSET[0] + x_mul * spacing),
                        float(VGC10_SUCTION_LOCAL_OFFSET[1] + y_mul * spacing),
                        float(VGC10_SUCTION_LOCAL_OFFSET[2]),
                    ))
                    set_prim_local_matrix(point_prim, point_mat)
                    # marker 자체가 충돌하지 않도록 명시적으로 물리 제거/비활성화
                    try:
                        if point_prim.HasAPI(UsdPhysics.CollisionAPI):
                            point_prim.RemoveAPI(UsdPhysics.CollisionAPI)
                    except Exception:
                        pass
                    try:
                        if point_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            point_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    except Exception:
                        pass
                    labels.append(label)
            print(
                f"  [SUCTION_GRID_CREATE] path={SUCTION_GRID_ROOT_PATH}, points={labels}, "
                f"spacing={spacing:.3f}m, total_width={2*spacing:.3f}m, visible={SUCTION_GRID_MARKERS_VISIBLE}"
            )
        except Exception as e:
            print(f"  [SUCTION_GRID_CREATE_FAIL] {e}")

    for _ in range(10):
        simulation_app.update()

    if VGC10_VISUAL_ONLY:
        make_subtree_visual_only(stage, VGC10_PRIM_PATH)

    update_vgc10_visual_follow_pose(stage, robot=None)

    try:
        omni.usd.get_context().get_selection().clear_selected_prim_paths()
    except Exception:
        pass

    print(f"  [OK] VGC10 root path  = {VGC10_PRIM_PATH}")
    print(f"  [OK] VGC10 mount path = {VGC10_MOUNT_PATH}")
    print(f"  [OK] VGC10 model path = {VGC10_MODEL_PATH}")
    print(f"       follow target   = {VGC10_FOLLOW_TARGET_PATH}")
    print(f"       local translate = {VGC10_LOCAL_TRANSLATE}")
    print(f"       local rotateXYZ = {VGC10_LOCAL_ROTATE_XYZ}")
    print(f"       local scale     = {VGC10_LOCAL_SCALE}")
    print(f"       suction point   = {VGC10_SUCTION_POINT_PATH}")
    print(f"       suction offset  = {VGC10_SUCTION_LOCAL_OFFSET}")


def _safe_token_from_path(path: str):
    token = str(path).strip("/").replace("/", "_")
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in token)


def _find_tool_target_for_robot_prim(stage, robot_prim_path: str):
    """robot_prim_path 아래에서 link_6/tool0를 찾는다. tool0가 없으면 link_6를 사용한다."""
    if not stage.GetPrimAtPath(robot_prim_path).IsValid():
        return None
    link6_path = find_prim_path_by_name(robot_prim_path, EE_LINK_NAME)
    if link6_path is None:
        return None
    tool0_path = link6_path + "/tool0"
    if stage.GetPrimAtPath(tool0_path).IsValid():
        return tool0_path
    return link6_path


def _resolve_robot_prim_from_root(stage, root_path: str):
    """/World/m0609_01/m0609 구조와 /World/m0609_01 자체 articulation 구조를 둘 다 허용한다."""
    candidates = [root_path + "/m0609", root_path]
    for p in candidates:
        if _find_tool_target_for_robot_prim(stage, p) is not None:
            return p
    return None


def _discover_idle_m0609_robot_prims(stage):
    """/World 아래 m0609_ 로 시작하는 active 외 로봇을 찾는다."""
    results = []
    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        return results
    for child in world.GetChildren():
        name = child.GetName()
        root_path = str(child.GetPath())
        if not name.startswith(IDLE_M0609_ROOT_PREFIX):
            continue
        if root_path == ACTIVE_ROBOT_ROOT_PATH:
            continue
        robot_path = _resolve_robot_prim_from_root(stage, root_path)
        if robot_path is not None:
            results.append(robot_path)
    return sorted(set(results))


def _create_vgc10_visual_for_target(stage, target_path: str, root_path: str, label: str):
    """idle 로봇에 active 로봇과 같은 VGC10 visual을 붙여 보이게 한다. 제어는 하지 않는다."""
    if target_path is None or not stage.GetPrimAtPath(target_path).IsValid():
        print(f"  [IDLE_VGC10_SKIP] target 없음: label={label}, target={target_path}")
        return False

    clear_prim_if_exists(stage, root_path)
    asset = _resolve_vgc10_asset_path()
    mount_path = root_path + "/vgc10_scaled_mount"
    model_path = mount_path + "/gripper_model"
    suction_path = root_path + "/vgc10_suction_point"

    root_prim = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    mount_prim = UsdGeom.Xform.Define(stage, mount_path).GetPrim()
    model_prim = UsdGeom.Xform.Define(stage, model_path).GetPrim()
    suction_prim = UsdGeom.Xform.Define(stage, suction_path).GetPrim()

    if Path(asset).exists():
        model_prim.GetReferences().AddReference(str(asset))
    else:
        print(f"  [IDLE_VGC10_WARN] VGC10 asset 없음: {asset}")
        UsdGeom.Cylinder.Define(stage, model_path + "/fallback_suction_pad")

    target_mat = get_world_matrix(stage, target_path)
    if target_mat is None:
        target_mat = Gf.Matrix4d(1.0)
    set_prim_local_matrix(root_prim, target_mat)

    mount_mat = make_trs_matrix(
        translate=VGC10_LOCAL_TRANSLATE,
        rotate_xyz_deg=VGC10_LOCAL_ROTATE_XYZ,
        scale=VGC10_LOCAL_SCALE,
    )
    set_prim_local_matrix(mount_prim, mount_mat)
    set_prim_local_matrix(model_prim, Gf.Matrix4d(1.0))

    suction_mat = Gf.Matrix4d(1.0)
    suction_mat.SetTranslateOnly(Gf.Vec3d(
        float(VGC10_SUCTION_LOCAL_OFFSET[0]),
        float(VGC10_SUCTION_LOCAL_OFFSET[1]),
        float(VGC10_SUCTION_LOCAL_OFFSET[2]),
    ))
    set_prim_local_matrix(suction_prim, suction_mat)

    for _ in range(10):
        simulation_app.update()
    if VGC10_VISUAL_ONLY:
        make_subtree_visual_only(stage, root_path)

    print(f"  [IDLE_VGC10_OK] {label}: root={root_path}")
    print(f"                  follow target={target_path}")
    return True


def attach_vgc10_to_idle_m0609_robots(stage):
    if not ATTACH_VGC10_TO_IDLE_M0609_ROBOTS:
        return []
    idle_robot_prims = _discover_idle_m0609_robot_prims(stage)
    attached = []
    if not idle_robot_prims:
        print("  [IDLE_VGC10] active 외 m0609_ 로봇을 찾지 못함")
        return attached
    for robot_path in idle_robot_prims:
        target = _find_tool_target_for_robot_prim(stage, robot_path)
        root_candidate = robot_path.rsplit("/m0609", 1)[0] if robot_path.endswith("/m0609") else robot_path
        token = _safe_token_from_path(root_candidate)
        vgc10_root = f"{IDLE_VGC10_ROOT_PREFIX}_{token}"
        if _create_vgc10_visual_for_target(stage, target, vgc10_root, label=root_candidate):
            attached.append((robot_path, vgc10_root, target))
    print(f"  [IDLE_VGC10_SUMMARY] idle robots with VGC10 visual = {len(attached)}")
    return attached

def zero_body_velocity(obj):
    try:
        obj.set_linear_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))
    except Exception:
        pass


class NullGripper:
    """
    기존 RG2 finger_joint를 삭제한 뒤에도 PickPlaceController가 gripper.forward()를
    호출할 수 있게 해주는 더미 그리퍼다. 실제 관절 명령은 만들지 않는다.
    """

    joint_opened_positions = np.array([], dtype=float)
    joint_closed_positions = np.array([], dtype=float)

    def initialize(self, *args, **kwargs):
        return None

    def reset(self):
        return None

    def set_joint_positions(self, *args, **kwargs):
        return None

    def get_joint_positions(self):
        return np.array([], dtype=float)

    def forward(self, action=None):
        return ArticulationAction(
            joint_positions=np.array([], dtype=float),
            joint_indices=np.array([], dtype=np.int32),
        )

    def open(self):
        return self.forward(action="open")

    def close(self):
        return self.forward(action="close")


def update_vgc10_suction_anchor(robot):
    """
    VGC10 기준 흡착 좌표만 계산한다.
    기존 scripted_suction_body / scripted_suction_direction_marker visual은 더 이상 움직이거나 생성하지 않는다.
    """
    stage = omni.usd.get_context().get_stage()

    # 먼저 VGC10 visual-follow root를 tool0 위치/자세로 갱신한다.
    update_vgc10_visual_follow_pose(stage, robot=robot)

    suction_pos = get_world_translation(stage, VGC10_SUCTION_POINT_PATH)
    if suction_pos is None:
        # fallback: VGC10 suction point를 못 찾으면 기존 EE 기준으로 계산
        ee_pos, _ = robot.end_effector.get_world_pose()
        suction_pos = np.array(ee_pos, dtype=float) + SUCTION_OFFSET_FROM_EE

    suction_pos = np.array(suction_pos, dtype=float)
    suction_pos[2] = max(float(suction_pos[2]), float(SUCTION_MIN_Z))

    # 디버그가 필요할 때만 작은 녹색 마커를 보여준다.
    if DEBUG_SHOW_SUCTION_POINT:
        marker_prim = stage.GetPrimAtPath(VGC10_SUCTION_DEBUG_MARKER_PATH)
        if not marker_prim.IsValid():
            marker = UsdGeom.Sphere.Define(stage, VGC10_SUCTION_DEBUG_MARKER_PATH)
            marker.CreateRadiusAttr(0.015)
            marker_prim = marker.GetPrim()
        marker_mat = Gf.Matrix4d(1.0)
        marker_mat.SetTranslateOnly(Gf.Vec3d(float(suction_pos[0]), float(suction_pos[1]), float(suction_pos[2])))
        set_prim_local_matrix(marker_prim, marker_mat)
    else:
        clear_prim_if_exists(stage, VGC10_SUCTION_DEBUG_MARKER_PATH)

    return suction_pos


def _sdf_path(path):
    return Sdf.Path(str(path))


def _gf_vec3_from_np(v):
    v = np.array(v, dtype=float)
    return Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))


def _world_to_local_point(stage, prim_path, world_point):
    mat = get_world_matrix(stage, prim_path)
    if mat is None:
        return np.zeros(3, dtype=float)
    try:
        inv = mat.GetInverse()
        wp = np.array(world_point, dtype=float)
        lp = inv.Transform(Gf.Vec3d(float(wp[0]), float(wp[1]), float(wp[2])))
        return np.array([float(lp[0]), float(lp[1]), float(lp[2])], dtype=float)
    except Exception:
        return np.zeros(3, dtype=float)




def _local_to_world_point(stage, prim_path, local_point):
    mat = get_world_matrix(stage, prim_path)
    if mat is None:
        return np.zeros(3, dtype=float)
    try:
        lp = np.array(local_point, dtype=float)
        wp = mat.Transform(Gf.Vec3d(float(lp[0]), float(lp[1]), float(lp[2])))
        return np.array([float(wp[0]), float(wp[1]), float(wp[2])], dtype=float)
    except Exception:
        return np.zeros(3, dtype=float)


def _fmt_vec(v, prec=4):
    try:
        v = np.array(v, dtype=float)
        return "(" + ",".join([f"{x:.{prec}f}" for x in v[:3]]) + ")"
    except Exception:
        return "(nan,nan,nan)"


def _safe_attr_value(prim, attr_name, default="<missing>"):
    try:
        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            v = attr.Get()
            return v if v is not None else "<None>"
    except Exception:
        pass
    return default


def _has_api_name(prim, api_cls):
    try:
        return bool(prim.HasAPI(api_cls))
    except Exception:
        return False


def _physics_diag_inspect_body(stage, path, label):
    prim = stage.GetPrimAtPath(str(path))
    print(f"[PHYSICS_DIAG_BODY][{label}] path={path}, valid={bool(prim and prim.IsValid())}")
    if not prim or not prim.IsValid():
        return
    pos = get_world_translation(stage, str(path))
    bbox = get_world_bbox_info(stage, str(path))
    print(
        f"[PHYSICS_DIAG_BODY][{label}] world_pos={_fmt_vec(pos)}, "
        f"RigidBodyAPI={_has_api_name(prim, UsdPhysics.RigidBodyAPI)}, "
        f"CollisionAPI={_has_api_name(prim, UsdPhysics.CollisionAPI)}, "
        f"MassAPI={_has_api_name(prim, UsdPhysics.MassAPI)}, "
        f"physics:rigidBodyEnabled={_safe_attr_value(prim, 'physics:rigidBodyEnabled')}, "
        f"physics:kinematicEnabled={_safe_attr_value(prim, 'physics:kinematicEnabled')}, "
        f"physics:collisionEnabled={_safe_attr_value(prim, 'physics:collisionEnabled')}"
    )
    print(
        f"[PHYSICS_DIAG_MASS][{label}] "
        f"mass={_safe_attr_value(prim, 'physics:mass')}, "
        f"density={_safe_attr_value(prim, 'physics:density')}, "
        f"centerOfMass={_safe_attr_value(prim, 'physics:centerOfMass')}, "
        f"diagonalInertia={_safe_attr_value(prim, 'physics:diagonalInertia')}, "
        f"principalAxes={_safe_attr_value(prim, 'physics:principalAxes')}"
    )
    if bbox is not None:
        print(
            f"[PHYSICS_DIAG_BBOX][{label}] center={_fmt_vec(bbox['center'])}, top={_fmt_vec(bbox['top_center'])}, "
            f"min={_fmt_vec(bbox['min'])}, max={_fmt_vec(bbox['max'])}, size={_fmt_vec(bbox['size'])}, height={bbox['height']:.4f}"
        )


def _physics_diag_inspect_colliders(stage, root_path, label="box", max_rows=60):
    root = stage.GetPrimAtPath(str(root_path))
    if not root or not root.IsValid():
        print(f"[PHYSICS_DIAG_COLLIDERS][{label}] root invalid: {root_path}")
        return
    rows = []
    for prim in Usd.PrimRange(root):
        try:
            has_col = prim.HasAPI(UsdPhysics.CollisionAPI) or bool(prim.GetAttribute("physics:collisionEnabled"))
            is_geom = _prim_is_geometry_like(prim)
            has_mesh_col = prim.HasAPI(UsdPhysics.MeshCollisionAPI) or bool(prim.GetAttribute("physics:approximation"))
            if has_col or is_geom or has_mesh_col:
                rows.append((str(prim.GetPath()), prim.GetTypeName(), has_col, has_mesh_col, _safe_attr_value(prim, "physics:collisionEnabled"), _safe_attr_value(prim, "physics:approximation")))
        except Exception as e:
            rows.append((str(prim.GetPath()), "<err>", False, False, f"err={e}", "<err>"))
    print(f"[PHYSICS_DIAG_COLLIDERS][{label}] count={len(rows)}, root={root_path}")
    for i, row in enumerate(rows[:max_rows]):
        p, typ, has_col, has_mesh_col, col_enabled, approx = row
        print(f"  [COLLIDER_ROW {i:02d}] path={p}, type={typ}, CollisionAPI={has_col}, MeshCollisionAPI={has_mesh_col}, enabled={col_enabled}, approximation={approx}")
    if len(rows) > max_rows:
        print(f"  [COLLIDER_ROW] ... truncated {len(rows)-max_rows} more rows")


def _create_or_set_attr(api_or_prim, attr_name, value):
    """UsdPhysics API attr를 생성/갱신한다. Isaac/USD 버전별 Create*Attr 차이를 피하기 위한 작은 helper."""
    try:
        # api_or_prim이 API 객체면 GetPrim이 있고, prim이면 그대로 쓴다.
        prim = api_or_prim.GetPrim() if hasattr(api_or_prim, "GetPrim") else api_or_prim
        attr = prim.GetAttribute(attr_name)
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
        attr.Set(value)
        return True
    except Exception:
        return False


def _physics_diag_apply_mass_inertia_104(stage, box_path):
    """104_: 원본 USD 저장 없이 현재 stage에서만 박스 MassAPI/inertia를 명시한다."""
    if not bool(globals().get("PHYSICS_MASS_INERTIA_DIAG_ENABLED", True)):
        print("[PHYSICS_MASS_INERTIA_104] disabled")
        return False

    prim = stage.GetPrimAtPath(str(box_path))
    if not prim or not prim.IsValid():
        print(f"[PHYSICS_MASS_INERTIA_104] box invalid: {box_path}")
        return False

    bbox = get_world_bbox_info(stage, str(box_path))
    root = get_world_translation(stage, str(box_path))
    if bbox is None or root is None:
        print(f"[PHYSICS_MASS_INERTIA_104] bbox/root unavailable: {box_path}")
        return False

    size = np.array(bbox.get("size", [0.26, 0.20, 0.24]), dtype=float)
    size = np.maximum(size, np.array([0.01, 0.01, 0.01], dtype=float))
    mass = float(globals().get("PHYSICS_MASS_INERTIA_DIAG_MASS", 2.0))
    scale = float(globals().get("PHYSICS_MASS_INERTIA_DIAG_INERTIA_SCALE", 1.0))

    # root 기준 local COM. 현재 OriBox root가 bbox min z에 가까우므로 보통 (0,0,0.12) 근처가 된다.
    try:
        com_local = _world_to_local_point(stage, str(box_path), np.array(bbox["center"], dtype=float))
    except Exception:
        com_local = np.array([0.0, 0.0, float(size[2]) * 0.5], dtype=float)

    hx, hy, hz = float(size[0]), float(size[1]), float(size[2])
    ixx = scale * (mass / 12.0) * (hy * hy + hz * hz)
    iyy = scale * (mass / 12.0) * (hx * hx + hz * hz)
    izz = scale * (mass / 12.0) * (hx * hx + hy * hy)

    print("\n========== [PHYSICS_MASS_INERTIA_APPLY_BEGIN_104] ==========")
    print(
        f"[PHYSICS_MASS_INERTIA_104][before] path={box_path}, "
        f"MassAPI={_has_api_name(prim, UsdPhysics.MassAPI)}, "
        f"mass={_safe_attr_value(prim, 'physics:mass')}, "
        f"centerOfMass={_safe_attr_value(prim, 'physics:centerOfMass')}, "
        f"diagonalInertia={_safe_attr_value(prim, 'physics:diagonalInertia')}, "
        f"principalAxes={_safe_attr_value(prim, 'physics:principalAxes')}"
    )
    print(
        f"[PHYSICS_MASS_INERTIA_104][computed] bbox_size={_fmt_vec(size)}, root={_fmt_vec(root)}, "
        f"bbox_center={_fmt_vec(bbox['center'])}, com_local={_fmt_vec(com_local)}, "
        f"mass={mass:.4f}, inertia=({ixx:.6f},{iyy:.6f},{izz:.6f}), scale={scale:.3f}"
    )

    try:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        try:
            mass_api.CreateMassAttr(float(mass)).Set(float(mass))
        except Exception:
            prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(float(mass))
        try:
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(float(com_local[0]), float(com_local[1]), float(com_local[2]))).Set(Gf.Vec3f(float(com_local[0]), float(com_local[1]), float(com_local[2])))
        except Exception:
            prim.CreateAttribute("physics:centerOfMass", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(float(com_local[0]), float(com_local[1]), float(com_local[2])))
        try:
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(float(ixx), float(iyy), float(izz))).Set(Gf.Vec3f(float(ixx), float(iyy), float(izz)))
        except Exception:
            prim.CreateAttribute("physics:diagonalInertia", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(float(ixx), float(iyy), float(izz)))
        try:
            mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0)).Set(Gf.Quatf(1.0))
        except Exception:
            prim.CreateAttribute("physics:principalAxes", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(1.0))
    except Exception as e:
        print(f"[PHYSICS_MASS_INERTIA_104][ERROR] apply failed: {e}")
        print("========== [PHYSICS_MASS_INERTIA_APPLY_FAIL_104] ==========\n")
        return False

    print(
        f"[PHYSICS_MASS_INERTIA_104][after] path={box_path}, "
        f"MassAPI={_has_api_name(prim, UsdPhysics.MassAPI)}, "
        f"mass={_safe_attr_value(prim, 'physics:mass')}, "
        f"centerOfMass={_safe_attr_value(prim, 'physics:centerOfMass')}, "
        f"diagonalInertia={_safe_attr_value(prim, 'physics:diagonalInertia')}, "
        f"principalAxes={_safe_attr_value(prim, 'physics:principalAxes')}"
    )
    print("========== [PHYSICS_MASS_INERTIA_APPLY_END_104] ==========\n")
    return True




# 105_: FixedJoint는 위치뿐 아니라 회전도 묶는다.
# localPos0/localPos1 위치 anchor는 103/104에서 0으로 맞는 것이 확인됐지만,
# localRot0/localRot1을 identity로 두면 body0(link_6)와 body1(box)의 joint frame 방향이 서로 다를 수 있다.
# 그래서 현재 body1(box)의 world orientation을 joint world orientation으로 삼고,
# body0/body1 각각의 localRot을 계산해 joint 생성 직전 두 joint frame 방향을 일치시킨다.
def _physics_diag_rot3_from_world_matrix_105(stage, prim_path):
    mat = get_world_matrix(stage, prim_path)
    if mat is None:
        return np.eye(3, dtype=float)
    cols = []
    for axis in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
        try:
            v = mat.TransformDir(Gf.Vec3d(float(axis[0]), float(axis[1]), float(axis[2])))
            cols.append(np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float))
        except Exception:
            cols.append(np.array(axis, dtype=float))
    R = np.column_stack(cols)
    try:
        # scale/shear가 섞여도 회전 행렬만 뽑기 위해 직교화한다.
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1.0
            R = U @ Vt
    except Exception:
        R = np.eye(3, dtype=float)
    return R


def _physics_diag_quatf_from_rot3_105(R):
    R = np.array(R, dtype=float)
    try:
        tr = float(R[0, 0] + R[1, 1] + R[2, 2])
        if tr > 0.0:
            S = (tr + 1.0) ** 0.5 * 2.0
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        n = max((w*w + x*x + y*y + z*z) ** 0.5, 1.0e-12)
        w, x, y, z = w/n, x/n, y/n, z/n
        return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))
    except Exception:
        return Gf.Quatf(1.0)


def _physics_diag_rot_desc_105(R):
    R = np.array(R, dtype=float)
    return (
        f"x={_fmt_vec(R[:,0])}, y={_fmt_vec(R[:,1])}, z={_fmt_vec(R[:,2])}, "
        f"det={float(np.linalg.det(R)):.6f}"
    )


def _physics_diag_compute_joint_local_rots_105(stage, body0, body1):
    R0 = _physics_diag_rot3_from_world_matrix_105(stage, body0)
    R1 = _physics_diag_rot3_from_world_matrix_105(stage, body1)
    # box의 현재 방향을 joint world frame으로 채택한다.
    # 그러면 body1 localRot1은 거의 identity가 되고, box의 현재 up 방향을 보존하는 쪽으로 constraint가 생성된다.
    Rj = R1.copy()
    L0 = R0.T @ Rj
    L1 = R1.T @ Rj
    W0 = R0 @ L0
    W1 = R1 @ L1
    rot_err = float(np.linalg.norm(W0 - W1))
    q0 = _physics_diag_quatf_from_rot3_105(L0)
    q1 = _physics_diag_quatf_from_rot3_105(L1)
    print("[PHYSICS_DIAG_LOCALROT_105] strategy=preserve_body1_box_world_orientation")
    print(f"[PHYSICS_DIAG_ROT_WORLD_105][body0] {_physics_diag_rot_desc_105(R0)}")
    print(f"[PHYSICS_DIAG_ROT_WORLD_105][body1] {_physics_diag_rot_desc_105(R1)}")
    print(f"[PHYSICS_DIAG_ROT_LOCAL_105][localRot0] {_physics_diag_rot_desc_105(L0)}")
    print(f"[PHYSICS_DIAG_ROT_LOCAL_105][localRot1] {_physics_diag_rot_desc_105(L1)}")
    print(f"[PHYSICS_DIAG_ROT_CHECK_105] world_rot_err={rot_err:.9f}, body1_up_z_before={float(R1[2,2]):.6f}")
    return {"q0": q0, "q1": q1, "R0": R0, "R1": R1, "L0": L0, "L1": L1, "rot_err": rot_err}


def _physics_diag_body_up_z_105(stage, prim_path):
    R = _physics_diag_rot3_from_world_matrix_105(stage, prim_path)
    return float(R[2, 2]), R

_PHYSICS_JOINT_DIAG_STATE = {}

def _physics_diag_capture_pose(stage, body0, body1, local0, local1, attach_world, label="capture"):
    bbox = get_world_bbox_info(stage, body1)
    root = get_world_translation(stage, body1)
    anchor0 = _local_to_world_point(stage, body0, local0)
    anchor1 = _local_to_world_point(stage, body1, local1)
    suction = get_world_translation(stage, VGC10_SUCTION_POINT_PATH)
    body1_up_z, _body1_R = _physics_diag_body_up_z_105(stage, body1)
    body0_up_z, _body0_R = _physics_diag_body_up_z_105(stage, body0)
    root_above_bbox = None
    try:
        if root is not None and bbox is not None:
            root_above_bbox = float(root[2] - bbox["center"][2])
    except Exception:
        root_above_bbox = None
    return {
        "label": label,
        "body0": str(body0),
        "body1": str(body1),
        "local0": np.array(local0, dtype=float),
        "local1": np.array(local1, dtype=float),
        "attach_world": np.array(attach_world, dtype=float),
        "anchor0": anchor0,
        "anchor1": anchor1,
        "anchor_err": float(np.linalg.norm(anchor0 - anchor1)),
        "anchor0_attach_err": float(np.linalg.norm(anchor0 - np.array(attach_world, dtype=float))),
        "anchor1_attach_err": float(np.linalg.norm(anchor1 - np.array(attach_world, dtype=float))),
        "root": root,
        "bbox_center": bbox["center"] if bbox is not None else None,
        "bbox_top": bbox["top_center"] if bbox is not None else None,
        "bbox_size": bbox["size"] if bbox is not None else None,
        "suction": suction,
        "body0_up_z": body0_up_z,
        "body1_up_z": body1_up_z,
        "root_minus_bbox_center_z": root_above_bbox,
    }


def _physics_diag_print_capture(prefix, cap, base=None):
    if not cap:
        print(f"[PHYSICS_DIAG]{prefix} <no capture>")
        return
    root_delta = bbox_delta = top_delta = suction_delta = None
    if base:
        try:
            if cap.get("root") is not None and base.get("root") is not None:
                root_delta = float(np.linalg.norm(np.array(cap["root"]) - np.array(base["root"])))
            if cap.get("bbox_center") is not None and base.get("bbox_center") is not None:
                bbox_delta = float(np.linalg.norm(np.array(cap["bbox_center"]) - np.array(base["bbox_center"])))
            if cap.get("bbox_top") is not None and base.get("bbox_top") is not None:
                top_delta = float(np.linalg.norm(np.array(cap["bbox_top"]) - np.array(base["bbox_top"])))
            if cap.get("suction") is not None and base.get("suction") is not None:
                suction_delta = float(np.linalg.norm(np.array(cap["suction"]) - np.array(base["suction"])))
        except Exception:
            pass
    print(
        f"[PHYSICS_DIAG]{prefix} "
        f"attach={_fmt_vec(cap.get('attach_world'))}, anchor0={_fmt_vec(cap.get('anchor0'))}, anchor1={_fmt_vec(cap.get('anchor1'))}, "
        f"anchor_err={cap.get('anchor_err', float('nan')):.6f}, "
        f"a0_attach_err={cap.get('anchor0_attach_err', float('nan')):.6f}, a1_attach_err={cap.get('anchor1_attach_err', float('nan')):.6f}, "
        f"root={_fmt_vec(cap.get('root'))}, bbox_center={_fmt_vec(cap.get('bbox_center'))}, bbox_top={_fmt_vec(cap.get('bbox_top'))}, "
        f"suction={_fmt_vec(cap.get('suction'))}, "
        f"body1_up_z={cap.get('body1_up_z', float('nan')):.6f}, "
        f"root_minus_bbox_z={cap.get('root_minus_bbox_center_z') if cap.get('root_minus_bbox_center_z') is not None else float('nan'):.6f}, "
        f"d_root={root_delta if root_delta is not None else -1:.6f}, "
        f"d_bbox={bbox_delta if bbox_delta is not None else -1:.6f}, "
        f"d_top={top_delta if top_delta is not None else -1:.6f}, "
        f"d_suction={suction_delta if suction_delta is not None else -1:.6f}"
    )


def physics_diag_log_after_attach_step(stage, phase_name, step, force=False):
    global _PHYSICS_JOINT_DIAG_STATE
    st = _PHYSICS_JOINT_DIAG_STATE or {}
    if not st.get("active"):
        return
    sample_steps = set(globals().get("PHYSICS_DIAGNOSTIC_SAMPLE_STEPS", {0, 1, 2, 5, 10}))
    every_until = int(globals().get("PHYSICS_DIAGNOSTIC_LOG_EVERY_STEP_UNTIL", 5))
    if not force and not (int(step) <= every_until or int(step) in sample_steps):
        return
    cap = _physics_diag_capture_pose(stage, st["body0"], st["body1"], st["local0"], st["local1"], st["attach_world"], label=f"{phase_name}:{step}")
    _physics_diag_print_capture(f"[after_attach phase={phase_name} step={step}]", cap, base=st.get("before"))

def get_suction_grid_world_points(stage):
    """
    55_: 3x3 흡착점 world 좌표를 얻는다.

    기존 51~53_ 방식은 /World/vgc10_visual_follow/vgc10_suction_grid 하위 prim의
    실제 world pose를 읽었다. 그런데 gripper/tool0 방향이 하늘을 보거나 기울면
    9점 배열도 같이 기울어져 상자 윗면 판정이 불안정해졌다.

    이번 버전은 기본적으로 현재 suction 중심을 기준으로 world XY 평면에
    3x3 가상 흡착점을 만든다. 그래서 '상자 윗면을 제대로 덮는가'만 확인한다.
    """
    result = []

    center_pos = get_world_translation(stage, VGC10_SUCTION_POINT_PATH)
    if center_pos is None:
        center_pos = get_world_translation(stage, VGC10_PRIM_PATH)
    if center_pos is not None and bool(SUCTION_GRID_EVALUATE_AS_WORLD_XY_GRID):
        center_pos = np.array(center_pos, dtype=float)
        spacing = float(SUCTION_GRID_SPACING_XY)
        for iy, y_mul in enumerate([-1.0, 0.0, 1.0]):
            for ix, x_mul in enumerate([-1.0, 0.0, 1.0]):
                label = f"p{iy}{ix}"
                pos = np.array([
                    center_pos[0] + x_mul * spacing,
                    center_pos[1] + y_mul * spacing,
                    center_pos[2],
                ], dtype=float)
                result.append({
                    "label": label,
                    "path": f"{SUCTION_GRID_WORLD_MARKER_ROOT_PATH}/{label}",
                    "pos": pos,
                    "center": (label == "p11"),
                })

        # 화면에서도 확인할 수 있도록 world XY 기준 마커를 갱신한다.
        # 이 마커들은 물리 없음/판정용 표시 전용이다.
        if SUCTION_GRID_MARKERS_VISIBLE:
            try:
                root = UsdGeom.Xform.Define(stage, SUCTION_GRID_WORLD_MARKER_ROOT_PATH).GetPrim()
                set_prim_local_matrix(root, Gf.Matrix4d(1.0))
                for pt in result:
                    prim = stage.GetPrimAtPath(pt["path"])
                    if not prim.IsValid():
                        prim = UsdGeom.Sphere.Define(stage, pt["path"]).GetPrim()
                        try:
                            UsdGeom.Sphere(prim).CreateRadiusAttr(float(SUCTION_GRID_MARKER_RADIUS))
                            UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.6, 1.0)])
                        except Exception:
                            pass
                    mat = Gf.Matrix4d(1.0)
                    pp = np.array(pt["pos"], dtype=float)
                    mat.SetTranslateOnly(Gf.Vec3d(float(pp[0]), float(pp[1]), float(pp[2])))
                    set_prim_local_matrix(prim, mat)
                    try:
                        if prim.HasAPI(UsdPhysics.CollisionAPI):
                            prim.RemoveAPI(UsdPhysics.CollisionAPI)
                    except Exception:
                        pass
                    try:
                        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    except Exception:
                        pass
            except Exception as e:
                try:
                    print(f"  [SUCTION_WORLD_GRID_MARKER_FAIL] {e}")
                except Exception:
                    pass
        return result

    if SUCTION_GRID_ENABLED:
        for iy in range(3):
            for ix in range(3):
                label = f"p{iy}{ix}"
                path = f"{SUCTION_GRID_ROOT_PATH}/{label}"
                pos = get_world_translation(stage, path)
                if pos is not None:
                    result.append({"label": label, "path": path, "pos": np.array(pos, dtype=float), "center": (label == "p11")})
    if not result:
        pos = get_world_translation(stage, VGC10_SUCTION_POINT_PATH)
        if pos is not None:
            result.append({"label": "p11_fallback", "path": VGC10_SUCTION_POINT_PATH, "pos": np.array(pos, dtype=float), "center": True})
    return result


_LAST_SUCTION_GRID_INFO = {"attach_center": None, "hits": [], "summary": "not_evaluated"}


def evaluate_suction_grid_on_box_top(stage, bbox_info, event=None, verbose=False):
    """
    59_: 중심 흡착점 1개만 사용한다.
    9점 평균/부분 hit 방식은 상자 자세와 carry offset을 흔들 수 있어서 중지한다.

    판정 기준:
    - 중심 흡착점 p_center가 박스 윗면 중심 근처에 있어야 함
    - 중심 흡착점 z가 실제 box_top_z 근처여야 함
    - attach_center는 항상 실제 박스 윗면 z + eps로 투영
    """
    global _LAST_SUCTION_GRID_INFO
    if bbox_info is None:
        _LAST_SUCTION_GRID_INFO = {"attach_center": None, "hits": [], "summary": "no_bbox"}
        return False, "center_no_bbox", _LAST_SUCTION_GRID_INFO

    pos = get_world_translation(stage, VGC10_SUCTION_POINT_PATH)
    if pos is None:
        _LAST_SUCTION_GRID_INFO = {"attach_center": None, "hits": [], "summary": "no_center_suction_point"}
        return False, "center_no_suction_point", _LAST_SUCTION_GRID_INFO

    top_center = np.array(bbox_info["top_center"], dtype=float)
    size = np.array(bbox_info["size"], dtype=float)
    half_x = max(float(size[0]) * 0.5, 0.0)
    half_y = max(float(size[1]) * 0.5, 0.0)
    allowed_x = max(0.0, half_x - BOX_TOP_SURFACE_MARGIN_X)
    allowed_y = max(0.0, half_y - BOX_TOP_SURFACE_MARGIN_Y)

    pos = np.array(pos, dtype=float)
    dx = float(pos[0] - top_center[0])
    dy = float(pos[1] - top_center[1])
    xy_err = float(np.linalg.norm(pos[:2] - top_center[:2]))
    z_gap = float(pos[2] - top_center[2])

    inside_rect = (abs(dx) <= allowed_x) and (abs(dy) <= allowed_y)
    within_center_tol = xy_err <= float(CENTER_SUCTION_CENTER_TOL_XY)
    within_effective_radius = xy_err <= float(CENTER_SUCTION_EFFECTIVE_RADIUS_XY)
    z_ok = BOX_ATTACH_Z_MIN <= z_gap <= BOX_ATTACH_Z_MAX

    ok = bool(inside_rect and within_center_tol and within_effective_radius and z_ok)

    attach_center = np.array([pos[0], pos[1], top_center[2] + PHYSICS_ATTACH_TOP_SURFACE_EPS], dtype=float)
    # 중심 흡착 방식에서는 attach_center XY를 box top 중심 쪽으로 약간 끌어와서 offset이 한쪽으로 치우치지 않게 한다.
    attach_center[:2] = top_center[:2] + np.clip(pos[:2] - top_center[:2], -0.020, 0.020)

    blockers = []
    if not inside_rect:
        blockers.append("CENTER_OUTSIDE_TOP_RECT")
    if not within_center_tol:
        blockers.append(f"CENTER_XY_ERR({xy_err:.3f}>{CENTER_SUCTION_CENTER_TOL_XY:.3f})")
    if not within_effective_radius:
        blockers.append(f"RADIUS({xy_err:.3f}>{CENTER_SUCTION_EFFECTIVE_RADIUS_XY:.3f})")
    if not z_ok:
        blockers.append(f"Z_GAP({z_gap:.3f} not in [{BOX_ATTACH_Z_MIN:.3f},{BOX_ATTACH_Z_MAX:.3f}])")

    summary = (
        f"center_mode=True,ok={ok},"
        f"dx={dx:+.4f},dy={dy:+.4f},xy_err={xy_err:.4f},"
        f"z_gap={z_gap:+.4f},z_gap_ok=[{BOX_ATTACH_Z_MIN:.3f},{BOX_ATTACH_Z_MAX:.3f}],"
        f"radius={CENTER_SUCTION_EFFECTIVE_RADIUS_XY:.3f},center_tol={CENTER_SUCTION_CENTER_TOL_XY:.3f},"
        f"top=({top_center[0]:.3f},{top_center[1]:.3f},{top_center[2]:.4f}),"
        f"allowed_rect=({allowed_x:.4f},{allowed_y:.4f}),"
        f"attach_center=({attach_center[0]:.3f},{attach_center[1]:.3f},{attach_center[2]:.4f}),"
        f"blockers={'|'.join(blockers) if blockers else 'NONE'}"
    )

    hit_infos = []
    if ok:
        hit_infos.append({"label": "p_center", "pos": pos, "dx": dx, "dy": dy, "z_gap": z_gap})

    point_logs = [
        f"p_center:hit={int(ok)},dx={dx:+.3f},dy={dy:+.3f},xy={xy_err:.3f},zgap={z_gap:+.3f},radius={CENTER_SUCTION_EFFECTIVE_RADIUS_XY:.3f}"
    ]

    _LAST_SUCTION_GRID_INFO = {
        "attach_center": attach_center,
        "hits": hit_infos,
        "hit_count": 1 if ok else 0,
        "center_ok": ok,
        "summary": summary,
        "point_logs": point_logs,
        "ok": ok,
    }

    try:
        evaluate_suction_grid_on_box_top._counter += 1
    except Exception:
        evaluate_suction_grid_on_box_top._counter = 1
    counter = int(evaluate_suction_grid_on_box_top._counter)
    do_log = bool(verbose or ok or SUCTION_GRID_LOG_EVERY_STEP or counter % int(max(1, CENTER_SUCTION_LOG_INTERVAL)) == 0)
    if do_log:
        print(f"  [CENTER_SUCTION] event={event}, {summary}")
        if SUCTION_GRID_VERBOSE_POINTS:
            print("                 " + " | ".join(point_logs))

    return ok, summary, _LAST_SUCTION_GRID_INFO

def _find_physics_body0_for_attach(stage):
    # link_6를 우선 사용한다. tool0는 Xform일 가능성이 높다.
    link_path = find_prim_path_by_name(ROBOT_PRIM_PATH, PHYSICS_ATTACH_BODY0_LINK_NAME)
    candidates = []
    if link_path:
        candidates.append(link_path)
    candidates.extend([VGC10_FOLLOW_TARGET_PATH, ROBOT_PRIM_PATH])
    for path in candidates:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            return path
    return None


def create_physics_attach_joint(stage, box_body_path, attach_world_point):
    """105_: MassAPI/inertia를 유지하고 FixedJoint localRot0/localRot1을 현재 body 자세 기준으로 정렬한 뒤 진단한다."""
    global _PHYSICS_JOINT_DIAG_STATE
    if not PHYSICS_FIXED_JOINT_ATTACH_ENABLED:
        return False
    clear_prim_if_exists(stage, PHYSICS_ATTACH_JOINT_PATH)

    body0 = _find_physics_body0_for_attach(stage)
    body1 = str(box_body_path)
    if body0 is None or not stage.GetPrimAtPath(body1).IsValid():
        print(f"  [PHYSICS_ATTACH_FAIL] body0={body0}, body1={body1}")
        return False

    print("\n========== [PHYSICS_DIAG_ATTACH_BEGIN_105] ==========")
    print(f"[PHYSICS_DIAG_INPUT] body0={body0}, body1={body1}, requested_attach_world={_fmt_vec(attach_world_point)}")
    _physics_diag_inspect_body(stage, body0, "body0_link")
    _physics_diag_inspect_body(stage, body1, "body1_box")
    if bool(globals().get("PHYSICS_DIAGNOSTIC_LOG_COLLIDER_TREE", True)):
        _physics_diag_inspect_colliders(stage, body1, "body1_box")

    # 104_: 원인 분리를 위해 FixedJoint 생성 전에 박스 mass / centerOfMass / inertia를 현재 stage에만 임시 명시한다.
    _physics_diag_apply_mass_inertia_104(stage, body1)
    _physics_diag_inspect_body(stage, body1, "body1_box_after_mass_inertia_104")

    # 박스 rigid/collision이 없으면 적용한다. 단, 비활성화하지 않는다.
    try:
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(body1))
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(body1)).CreateCollisionEnabledAttr(True)
    except Exception as e:
        print(f"  [PHYSICS_ATTACH_WARN] body1 physics apply warning: {e}")

    wp = np.array(attach_world_point, dtype=float)
    local0 = _world_to_local_point(stage, body0, wp)
    local1 = _world_to_local_point(stage, body1, wp)
    anchor0_pre = _local_to_world_point(stage, body0, local0)
    anchor1_pre = _local_to_world_point(stage, body1, local1)
    before_cap = _physics_diag_capture_pose(stage, body0, body1, local0, local1, wp, label="before_joint_define")
    _physics_diag_print_capture("[before_joint_define]", before_cap)
    print(
        f"[PHYSICS_DIAG_LOCAL] local0={_fmt_vec(local0)}, local1={_fmt_vec(local1)}, "
        f"pre_anchor0={_fmt_vec(anchor0_pre)}, pre_anchor1={_fmt_vec(anchor1_pre)}, "
        f"pre_anchor_err={float(np.linalg.norm(anchor0_pre-anchor1_pre)):.6f}"
    )
    # 105_: 위치 anchor는 이미 맞는 것으로 확인됐으므로, 이번에는 회전 joint frame을 현재 box 방향 기준으로 맞춘다.
    rot_diag_105 = _physics_diag_compute_joint_local_rots_105(stage, body0, body1)

    try:
        joint = UsdPhysics.FixedJoint.Define(stage, PHYSICS_ATTACH_JOINT_PATH)
        joint.CreateBody0Rel().SetTargets([_sdf_path(body0)])
        joint.CreateBody1Rel().SetTargets([_sdf_path(body1)])
        joint.CreateLocalPos0Attr(_gf_vec3_from_np(local0))
        joint.CreateLocalPos1Attr(_gf_vec3_from_np(local1))
        try:
            joint.CreateLocalRot0Attr(rot_diag_105.get("q0", Gf.Quatf(1.0)))
            joint.CreateLocalRot1Attr(rot_diag_105.get("q1", Gf.Quatf(1.0)))
        except Exception as e:
            print(f"  [PHYSICS_LOCALROT_WARN_105] failed to set localRot attrs: {e}")
        print(
            f"  [PHYSICS_ATTACH_OK] joint={PHYSICS_ATTACH_JOINT_PATH}, body0={body0}, body1={body1}, "
            f"world=({wp[0]:.3f},{wp[1]:.3f},{wp[2]:.4f}), "
            f"local0=({local0[0]:.3f},{local0[1]:.3f},{local0[2]:.3f}), "
            f"local1=({local1[0]:.3f},{local1[1]:.3f},{local1[2]:.3f}), "
            f"localRotMode=preserve_body1_box_world_orientation, rot_err={rot_diag_105.get('rot_err', -1.0):.9f}"
        )
        after_define_cap = _physics_diag_capture_pose(stage, body0, body1, local0, local1, wp, label="after_joint_define_before_step")
        _physics_diag_print_capture("[after_joint_define_before_step]", after_define_cap, base=before_cap)
        _PHYSICS_JOINT_DIAG_STATE = {
            "active": True,
            "body0": body0,
            "body1": body1,
            "local0": np.array(local0, dtype=float),
            "local1": np.array(local1, dtype=float),
            "attach_world": np.array(wp, dtype=float),
            "localRot0": rot_diag_105.get("q0", Gf.Quatf(1.0)),
            "localRot1": rot_diag_105.get("q1", Gf.Quatf(1.0)),
            "rot_diag_105": rot_diag_105,
            "before": before_cap,
            "after_define": after_define_cap,
        }
        print("========== [PHYSICS_DIAG_ATTACH_END_105] ==========\n")
        return True
    except Exception as e:
        print(f"  [PHYSICS_ATTACH_FAIL] create joint error: {e}")
        print("========== [PHYSICS_DIAG_ATTACH_FAIL_105] ==========\n")
        return False

def release_physics_attach_joint(stage, reason="release"):
    if stage.GetPrimAtPath(PHYSICS_ATTACH_JOINT_PATH).IsValid():
        try:
            stage.RemovePrim(PHYSICS_ATTACH_JOINT_PATH)
            print(f"  [PHYSICS_RELEASE_OK] joint removed: {PHYSICS_ATTACH_JOINT_PATH}, reason={reason}")
            return True
        except Exception as e:
            print(f"  [PHYSICS_RELEASE_FAIL] {e}")
    return False


def _diag_bool_attr_128(prim, attr_name):
    try:
        attr = prim.GetAttribute(attr_name)
        if attr:
            return attr.Get()
    except Exception:
        pass
    return None


def _diag_set_bool_attr_128(prim, attr_name, value):
    try:
        attr = prim.GetAttribute(attr_name)
        if not attr:
            attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Bool, custom=False)
        attr.Set(bool(value))
        return True
    except Exception:
        return False


def _diag_set_vec3_attr_128(prim, attr_name, value):
    try:
        attr = prim.GetAttribute(attr_name)
        if attr:
            attr.Set(Gf.Vec3f(float(value[0]), float(value[1]), float(value[2])))
            return True
    except Exception:
        pass
    return False


def _diag_has_attr_128(prim, *names):
    for name in names:
        try:
            if prim.HasAttribute(name) or prim.GetAttribute(name):
                return True
        except Exception:
            pass
    return False


def _diag_attr_names_with_128(prim, text):
    out = []
    try:
        low = str(text).lower()
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if low in name.lower():
                out.append(name)
    except Exception:
        pass
    return out


def _diag_rel_targets_128(prim, rel_name):
    try:
        rel = prim.GetRelationship(rel_name)
        if rel:
            return [str(t) for t in rel.GetTargets()]
    except Exception:
        pass
    return []


def scan_stage_joints_128(stage, box_root_path, label="scan"):
    """Stage 전체 joint를 짧게 스캔해서 box/root/vgc10/m0609와 연결된 joint가 남았는지 확인."""
    if not bool(globals().get("RELEASE_DIAG_SCAN_ALL_JOINTS_128", True)):
        return []
    box_root_path = str(box_root_path)
    rows = []
    try:
        for prim in stage.Traverse():
            try:
                typ = str(prim.GetTypeName())
                path = str(prim.GetPath())
                has_body_rel = bool(prim.GetRelationship("physics:body0")) or bool(prim.GetRelationship("physics:body1"))
                if ("Joint" not in typ) and ("joint" not in path.lower()) and (not has_body_rel):
                    continue
                b0 = _diag_rel_targets_128(prim, "physics:body0")
                b1 = _diag_rel_targets_128(prim, "physics:body1")
                joined = " ".join([path, typ] + b0 + b1)
                relevant = (
                    box_root_path in joined
                    or "/OriBox" in joined
                    or "vgc10" in joined.lower()
                    or "m0609_A" in joined
                    or path == str(PHYSICS_ATTACH_JOINT_PATH)
                )
                if relevant:
                    rows.append((path, typ, b0, b1))
            except Exception:
                pass
    except Exception as exc:
        print(f"[DIAG128_JOINT_SCAN_WARN][{label}] {type(exc).__name__}: {exc}")
        return []

    print(f"[DIAG128_JOINT_SCAN][{label}] relevant_joint_count={len(rows)}")
    max_rows = int(globals().get("RELEASE_DIAG_MAX_ROWS_128", 24))
    for i, (path, typ, b0, b1) in enumerate(rows[:max_rows]):
        print(f"  [J{i:02d}] path={path}, type={typ}, body0={b0}, body1={b1}")
    if len(rows) > max_rows:
        print(f"  [J...] truncated={len(rows) - max_rows}")
    return rows


def compact_box_physics_state_128(stage, root_path, label="state"):
    """상자 root/subtree의 physics 상태를 핵심만 출력."""
    root_path = str(root_path)
    root = stage.GetPrimAtPath(root_path)
    print(f"[DIAG128_BODY_STATE][{label}] root={root_path}, valid={bool(root and root.IsValid())}")
    if not root or not root.IsValid():
        return []

    bbox = get_world_bbox_info(stage, root_path)
    if bbox is not None:
        c = np.array(bbox["center"], dtype=float)
        mn = np.array(bbox["min"], dtype=float)
        mx = np.array(bbox["max"], dtype=float)
        print(
            f"[DIAG128_BBOX][{label}] center=({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f}), "
            f"bottom_z={mn[2]:+.4f}, top_z={mx[2]:+.4f}, height={float(mx[2]-mn[2]):.4f}"
        )

    rows = []
    for prim in Usd.PrimRange(root):
        try:
            path = str(prim.GetPath())
            is_root = path == root_path
            rb_api = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            col_api = prim.HasAPI(UsdPhysics.CollisionAPI)
            mass_api = prim.HasAPI(UsdPhysics.MassAPI)
            has_physics_attr = any(str(a.GetName()).startswith("physics:") or str(a.GetName()).startswith("physx") for a in prim.GetAttributes())
            if not (is_root or rb_api or col_api or mass_api or has_physics_attr):
                continue
            gravity_attrs = _diag_attr_names_with_128(prim, "gravity")
            rows.append({
                "path": path,
                "type": str(prim.GetTypeName()),
                "rb_api": rb_api,
                "col_api": col_api,
                "mass_api": mass_api,
                "rb_enabled": _diag_bool_attr_128(prim, "physics:rigidBodyEnabled"),
                "kinematic": _diag_bool_attr_128(prim, "physics:kinematicEnabled"),
                "collision": _diag_bool_attr_128(prim, "physics:collisionEnabled"),
                "velocity": _safe_attr_value(prim, "physics:velocity"),
                "angular_velocity": _safe_attr_value(prim, "physics:angularVelocity"),
                "gravity_attrs": {name: _safe_attr_value(prim, name) for name in gravity_attrs[:6]},
            })
        except Exception:
            pass

    print(f"[DIAG128_BODY_ROWS][{label}] rows={len(rows)}")
    max_rows = int(globals().get("RELEASE_DIAG_MAX_ROWS_128", 24))
    for i, r in enumerate(rows[:max_rows]):
        print(
            f"  [B{i:02d}] path={r['path']}, type={r['type']}, "
            f"RBAPI={r['rb_api']}, rbEnabled={r['rb_enabled']}, kin={r['kinematic']}, "
            f"COLAPI={r['col_api']}, colEnabled={r['collision']}, gravity={r['gravity_attrs']}"
        )
    if len(rows) > max_rows:
        print(f"  [B...] truncated={len(rows) - max_rows}")
    return rows


def force_box_dynamic_after_release_128(stage, root_path, verbose=True):
    """release 직후에만 호출. 중간 운반 중에는 호출 금지."""
    root_path = str(root_path)
    root = stage.GetPrimAtPath(root_path)
    stats = {"rb_apply": 0, "rb_on": 0, "kin_off": 0, "col_on": 0, "gravity_off": 0, "vel_zero": 0}
    if not root or not root.IsValid():
        print(f"[DIAG128_DYNAMIC_RESTORE_FAIL] invalid root={root_path}")
        return stats

    # root는 반드시 rigid body 후보로 본다. child는 기존에 rigid 관련 흔적이 있는 경우만 건드린다.
    for prim in Usd.PrimRange(root):
        try:
            path = str(prim.GetPath())
            is_root = path == root_path
            has_rb_hint = (
                is_root
                or prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or _diag_has_attr_128(prim, "physics:rigidBodyEnabled", "physics:kinematicEnabled", "physics:velocity", "physics:angularVelocity")
            )
            if has_rb_hint:
                try:
                    UsdPhysics.RigidBodyAPI.Apply(prim)
                    stats["rb_apply"] += 1
                except Exception:
                    pass
                if _diag_set_bool_attr_128(prim, "physics:rigidBodyEnabled", True):
                    stats["rb_on"] += 1
                if _diag_set_bool_attr_128(prim, "physics:kinematicEnabled", False):
                    stats["kin_off"] += 1
                for gname in _diag_attr_names_with_128(prim, "gravity"):
                    # disableGravity 계열이면 False가 중력 ON이다.
                    if "disable" in gname.lower() and _diag_set_bool_attr_128(prim, gname, False):
                        stats["gravity_off"] += 1
                if bool(globals().get("RELEASE_DIAG_ZERO_VELOCITY_128", True)):
                    if _diag_set_vec3_attr_128(prim, "physics:velocity", (0, 0, 0)):
                        stats["vel_zero"] += 1
                    if _diag_set_vec3_attr_128(prim, "physics:angularVelocity", (0, 0, 0)):
                        stats["vel_zero"] += 1
        except Exception:
            pass

    if bool(globals().get("RELEASE_DIAG_RESTORE_COLLISION_128", True)):
        for prim in Usd.PrimRange(root):
            try:
                is_geom_like = _prim_is_geometry_like(prim)
                has_col_hint = prim.HasAPI(UsdPhysics.CollisionAPI) or _diag_has_attr_128(prim, "physics:collisionEnabled")
                if is_geom_like or has_col_hint:
                    try:
                        UsdPhysics.CollisionAPI.Apply(prim)
                    except Exception:
                        pass
                    if _diag_set_bool_attr_128(prim, "physics:collisionEnabled", True):
                        stats["col_on"] += 1
            except Exception:
                pass

    if verbose:
        print(f"[DIAG128_DYNAMIC_RESTORE] root={root_path}, stats={stats}")
    return stats


def compact_drop_observe_128(stage, root_path, label="observe"):
    bbox = get_world_bbox_info(stage, str(root_path))
    root_t = get_world_translation(stage, str(root_path))
    if bbox is None:
        print(f"[DIAG128_DROP_OBS][{label}] bbox=None, root_t={_fmt_vec(root_t)}")
        return
    c = np.array(bbox["center"], dtype=float)
    mn = np.array(bbox["min"], dtype=float)
    mx = np.array(bbox["max"], dtype=float)
    print(
        f"[DIAG128_DROP_OBS][{label}] root={_fmt_vec(root_t)}, "
        f"center=({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f}), bottom_z={mn[2]:+.4f}, top_z={mx[2]:+.4f}"
    )

def should_script_attach(event, suction_pos, cube_pos):
    """
    흡착 ON 판정.

    20_ 핵심 수정:
    19_의 gap_top 방식은 큐브 윗면 기준으로는 직관적이지만,
    현재 사용자가 직접 맞춘 SUCTION_HOLD_OFFSET=-0.056과 맞지 않았다.

    지금 큐브가 화면상 맞게 붙는 조건은:
        cube_center ≈ suction_pos + SUCTION_HOLD_OFFSET
    이다.

    그래서 이제는 hold_error를 본다.
        hold_error_z = (suction_z + SUCTION_HOLD_OFFSET_Z) - cube_z

    hold_error_z가 0에 가까우면, 지금 흡착해도 큐브가 튀거나 끌려 올라가지 않는다.
    hold_error_z가 음수면 VGC10이 이미 너무 내려간 상태라 큐브를 뚫는 방향이다.
    hold_error_z가 양수면 아직 멀어서 일찍 잡는 방향이다.
    """
    suction_pos = np.array(suction_pos, dtype=float)
    cube_pos = np.array(cube_pos, dtype=float)

    cube_top_z = float(cube_pos[2] + CUBE_HALF_Z)
    gap_top = float(suction_pos[2] - cube_top_z)

    dz = float(suction_pos[2] - cube_pos[2])
    xy = float(np.linalg.norm(suction_pos[:2] - cube_pos[:2]))
    dist = float(np.linalg.norm(suction_pos - cube_pos))

    held_target_pos = suction_pos + SUCTION_HOLD_OFFSET
    hold_error_vec = held_target_pos - cube_pos
    hold_error_z = float(hold_error_vec[2])
    hold_error_xy = float(np.linalg.norm(hold_error_vec[:2]))
    hold_error_dist = float(np.linalg.norm(hold_error_vec))

    metric = (
        f"hold_z={hold_error_z:.3f},hold_xy={hold_error_xy:.3f},hold_dist={hold_error_dist:.3f},"
        f"gap_top={gap_top:.3f},dz={dz:.3f},xy={xy:.3f},dist={dist:.3f}"
    )

    if event in PICK_CLOSE_EVENTS:
        # 1차 흡착 조건:
        # hold_z가 0 근처일 때만 붙인다.
        # hold_z < 0이면 이미 너무 내려간 상태라 큐브를 뚫는 방향이므로 금지한다.
        # hold_z > 0이 너무 크면 아직 멀어서 일찍 붙는 느낌이 나므로 금지한다.
        ok = (-0.003 <= hold_error_z <= 0.006) and (hold_error_xy <= 0.05) and (hold_error_dist <= 0.012)
        return ok, metric

    if event in RETRY_CLOSE_EVENTS:
        # 재시도 조건: 1차에서 약간 놓쳤을 때만 허용한다.
        # 너무 빨리 붙으면 오른쪽 0.010을 0.006으로 낮추고,
        # 뚫으면 왼쪽 -0.006을 0.000 쪽으로 올린다.
        ok = (-0.006 <= hold_error_z <= 0.010) and (hold_error_xy <= 0.05) and (hold_error_dist <= 0.016)
        return ok, "retry_" + metric

    return False, "not_pick_event"


# ============================================================
# Task
# ============================================================
# ============================================================
# Existing USD box utilities
# ============================================================
def resolve_first_valid_path(stage, paths):
    for p in paths:
        if stage.GetPrimAtPath(p).IsValid():
            return p
    return None


def ensure_oribox_a_exists(stage):  # legacy name: now also supports OriBoxB fallback
    """
    72_: Conveyor_lift.usd에서는 root(/World/OriBoxA_01)를 실제 박스 기준으로 사용한다.
    root가 이미 있으면 root path를 반환한다. fallback 생성 시에도 reference는 root에 붙인다.
    """
    if stage.GetPrimAtPath(BOX_ROOT_PATH).IsValid():
        print(f"  [OK] 기존 root 박스 사용: {BOX_ROOT_PATH}")
        return BOX_ROOT_PATH

    if not Path(ORI_BOX_USD_PATH).exists():
        raise RuntimeError(
            f"박스 root prim이 없고 oriA/oriB USD도 찾지 못했습니다.\n"
            f"  BOX_ROOT_PATH={BOX_ROOT_PATH}\n"
            f"  ORI_BOX_USD_PATH={ORI_BOX_USD_PATH}"
        )

    box_xform = UsdGeom.Xform.Define(stage, BOX_ROOT_PATH)
    box_prim = box_xform.GetPrim()
    box_prim.GetReferences().AddReference(str(ORI_BOX_USD_PATH))

    # fallback 초기 위치. 실제 Conveyor_lift.usd를 쓰면 보통 이 경로는 실행되지 않는다.
    _set_xform_common(
        box_prim,
        translate=BOX_SCREENSHOT_LOCAL_TRANSLATE,
        rotate=BOX_SCREENSHOT_LOCAL_ROTATE_XYZ,
        scale=BOX_SCREENSHOT_LOCAL_SCALE,
    )

    for _ in range(20):
        simulation_app.update()

    print(f"  [OK] OriBox fallback reference 생성(root): {BOX_ROOT_PATH}")
    print(f"       source = {ORI_BOX_USD_PATH}")
    return BOX_ROOT_PATH


def get_world_bbox_info(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None

    purposes = [
        UsdGeom.Tokens.default_,
        UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy,
        UsdGeom.Tokens.guide,
    ]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    try:
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn = aligned.GetMin()
        mx = aligned.GetMax()
        min_v = np.array([float(mn[0]), float(mn[1]), float(mn[2])], dtype=float)
        max_v = np.array([float(mx[0]), float(mx[1]), float(mx[2])], dtype=float)
        center = (min_v + max_v) * 0.5
        size = max_v - min_v
        top_center = np.array([center[0], center[1], max_v[2]], dtype=float)
        return {
            "min": min_v,
            "max": max_v,
            "center": center,
            "size": size,
            "top_center": top_center,
            "top_z": float(max_v[2]),
            "height": float(size[2]),
        }
    except Exception as exc:
        print(f"  [WARN] bbox 계산 실패: {prim_path} / {exc}")
        root_pos = get_world_translation(stage, prim_path)
        if root_pos is None:
            return None
        return {
            "min": root_pos.copy(),
            "max": root_pos.copy(),
            "center": root_pos.copy(),
            "size": np.zeros(3),
            "top_center": root_pos.copy(),
            "top_z": float(root_pos[2]),
            "height": 0.0,
        }




def log_oribox_pose_tracker(stage, label="pose", force=False):
    """
    64_ 디버그 전용.
    OriBoxA_02 / OriBoxA_03의 root world translate와 실제 Small_Cardboard_box bbox center/top을 계속 찍는다.
    - root_translate: 사용자가 말한 Prim의 Translate가 world에서 어디로 해석되는지 확인
    - bbox_center/top: 실제 박스 mesh가 어디에 있는지 확인
    - d_root/d_center: 직전 로그 대비 이동량. 갑자기 커지면 순간이동/재배치 발생 지점
    """
    if not bool(POSE_TRACK_ENABLED):
        return
    try:
        _POSE_TRACK_STATE["step"] = int(_POSE_TRACK_STATE.get("step", 0)) + 1
        step = int(_POSE_TRACK_STATE["step"])
        if (not force) and (not bool(POSE_TRACK_LOG_EVERY_STEP)) and (step % int(POSE_TRACK_LOG_INTERVAL) != 0):
            return

        prev = _POSE_TRACK_STATE.setdefault("prev", {})
        parts = []
        for root_path in POSE_TRACK_ROOT_PATHS:
            root_prim = stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                parts.append(f"{root_path}:MISSING")
                continue

            # 72_: 추적 기준도 root이다. child는 실제로 root에 붙어 따라오는지 확인용으로만 별도 출력한다.
            box_path = root_path
            child_path = find_named_child_or_descendant(stage, root_path, POSE_TRACK_BOX_CHILD_NAME)
            root_t = get_world_translation(stage, root_path)
            box_t = get_world_translation(stage, box_path)
            child_t = get_world_translation(stage, child_path) if child_path else None
            bbox = get_world_bbox_info(stage, root_path)

            root_t = np.array(root_t if root_t is not None else [np.nan, np.nan, np.nan], dtype=float)
            box_t = np.array(box_t if box_t is not None else [np.nan, np.nan, np.nan], dtype=float)
            if bbox is not None:
                center = np.array(bbox["center"], dtype=float)
                top = np.array(bbox["top_center"], dtype=float)
            else:
                center = np.array([np.nan, np.nan, np.nan], dtype=float)
                top = np.array([np.nan, np.nan, np.nan], dtype=float)

            old = prev.get(root_path)
            d_root = 0.0
            d_center = 0.0
            jump = ""
            if old is not None:
                try:
                    d_root = float(np.linalg.norm(root_t - old["root_t"]))
                    d_center = float(np.linalg.norm(center - old["center"]))
                    if max(d_root, d_center) >= float(POSE_TRACK_JUMP_WARN_TOL):
                        jump = f" JUMP>= {POSE_TRACK_JUMP_WARN_TOL:.3f}"
                except Exception:
                    pass

            prev[root_path] = {"root_t": root_t.copy(), "center": center.copy(), "box_t": box_t.copy(), "top": top.copy()}
            child_txt = ""
            try:
                if child_t is not None:
                    child_t_arr = np.array(child_t, dtype=float)
                    child_txt = f" child=({child_t_arr[0]:+.4f},{child_t_arr[1]:+.4f},{child_t_arr[2]:+.4f})"
            except Exception:
                child_txt = ""
            parts.append(
                f"{root_path}: root=({root_t[0]:+.4f},{root_t[1]:+.4f},{root_t[2]:+.4f}) "
                f"used_xform=({box_t[0]:+.4f},{box_t[1]:+.4f},{box_t[2]:+.4f}) "
                f"bbox_center=({center[0]:+.4f},{center[1]:+.4f},{center[2]:+.4f}) "
                f"bbox_top=({top[0]:+.4f},{top[1]:+.4f},{top[2]:+.4f}) "
                f"d_root={d_root:.4f} d_center={d_center:.4f}{jump}{child_txt}"
            )

        print(f"[POSE_TRACK][{label}][step={step}] " + " | ".join(parts))
    except Exception as exc:
        print(f"[POSE_TRACK_WARN][{label}] {exc}")


def _as_matrix4d(value):
    """USD Python 버전에 따라 GetLocalTransformation 반환 형태가 달라지는 것을 흡수한다."""
    if isinstance(value, tuple):
        value = value[0]
    try:
        return Gf.Matrix4d(value)
    except Exception:
        return Gf.Matrix4d(1.0)


def _get_or_create_scripted_transform_op(prim):
    """
    XformCommonAPI.SetTranslate가 실패하는 incompatible xformOp 스택을 피하기 위해
    강제로 단일 matrix transform op를 사용한다.

    로그에 아래 경고가 뜨면 이 방식이 필요하다.
      Could not determine xform ops for incompatible xformable
    """
    xformable = UsdGeom.Xformable(prim)
    attr_name = "xformOp:transform:scripted_carry"
    attr = prim.GetAttribute(attr_name)
    if attr and attr.IsValid():
        op = UsdGeom.XformOp(attr)
    else:
        try:
            op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "scripted_carry")
        except Exception:
            # 혹시 같은 이름 attribute가 이미 있는데 op wrapper 생성만 실패한 경우 fallback
            attr = prim.GetAttribute(attr_name)
            if attr and attr.IsValid():
                op = UsdGeom.XformOp(attr)
            else:
                raise

    # 기존 translate/rotate/scale op order가 incompatible이면 XformCommonAPI가 못 다루므로
    # scripted_carry matrix op 하나만 compose되게 만든다.
    try:
        xformable.SetXformOpOrder([op])
    except Exception:
        xformable.ClearXformOpOrder()
        xformable.SetXformOpOrder([op])
    return op


def set_prim_world_translation(stage, prim_path, world_pos, _sync_oribox_child=True):
    """
    prim을 원하는 world translation으로 보낸다.

    15_ 핵심:
    - XformCommonAPI.SetTranslate를 쓰지 않는다.
    - 기존 local rotate/scale을 matrix로 보존한 뒤 translation만 바꾼다.
    - /World/OriBoxA 또는 Small_Cardboard_box에서 뜨던 incompatible xformable 경고를 우회한다.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"  [WARN] set_prim_world_translation 실패: prim 없음 {prim_path}")
        return False

    world_pos = np.array(world_pos, dtype=float)

    try:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        parent = prim.GetParent()
        if parent and parent.IsValid():
            parent_world = cache.GetLocalToWorldTransform(parent)
            local_v = parent_world.GetInverse().Transform(
                Gf.Vec3d(float(world_pos[0]), float(world_pos[1]), float(world_pos[2]))
            )
        else:
            local_v = Gf.Vec3d(float(world_pos[0]), float(world_pos[1]), float(world_pos[2]))

        xformable = UsdGeom.Xformable(prim)
        local_mat = _as_matrix4d(xformable.GetLocalTransformation())
        # 기존 local rotation/scale은 보존하고 translate만 교체한다.
        local_mat.SetTranslateOnly(local_v)

        op = _get_or_create_scripted_transform_op(prim)
        op.Set(local_mat)

        # 68_ 핵심: OriBoxA_* root를 움직인 경우, 하위 Small_Cardboard_box의 world translate도
        # root와 같게 다시 맞춰 parent/child 중심 좌표가 벌어지지 않게 한다.
        try:
            if (
                _sync_oribox_child
                and bool(ORIBOX_KEEP_CHILD_CENTER_MATCHED_TO_ROOT)
                and any(str(prim_path).startswith(str(ORIBOX_STACK_PARENT_PATH).rstrip("/") + "/" + pref) for pref in ORIBOX_STACK_NAME_PREFIXES)
                and not str(prim_path).rstrip("/").endswith("/" + str(ORIBOX_STACK_BOX_MESH_NAME))
            ):
                child_path = str(prim_path).rstrip("/") + "/" + str(ORIBOX_STACK_BOX_MESH_NAME)
                child_prim = stage.GetPrimAtPath(child_path)
                if child_prim and child_prim.IsValid():
                    child_pos = get_world_translation(stage, child_path)
                    root_pos = np.array(world_pos, dtype=float)
                    child_delta = None if child_pos is None else float(np.linalg.norm(np.array(child_pos, dtype=float) - root_pos))
                    if child_delta is None or child_delta > float(ORIBOX_CENTER_MATCH_TOL):
                        set_prim_world_translation(stage, child_path, root_pos, _sync_oribox_child=False)
                        if bool(ORIBOX_CENTER_MATCH_LOG):
                            print(
                                f"  [ORIBOX_CENTER_SYNC] root={prim_path} child={child_path} "
                                f"delta_before={child_delta} -> child world translate matched to root"
                            )
        except Exception as sync_exc:
            print(f"  [WARN] ORIBOX_CENTER_SYNC 실패: {prim_path} / {sync_exc}")

        return True
    except Exception as exc:
        print(f"  [WARN] set_prim_world_translation matrix 실패: {prim_path} / {exc}")
        return False


def sync_oribox_child_centers_with_roots(stage, roots=None, verbose=True):
    """68_: USD에서 맞춘 OriBoxA_* root와 Small_Cardboard_box의 world translate 중심을 실행 중에도 맞춘다."""
    if not bool(ORIBOX_KEEP_CHILD_CENTER_MATCHED_TO_ROOT):
        return

    if roots is None:
        roots = []
        try:
            parent = stage.GetPrimAtPath(ORIBOX_STACK_PARENT_PATH)
            if parent and parent.IsValid():
                for child in parent.GetChildren():
                    if child.GetName().startswith(ORIBOX_STACK_NAME_PREFIXES):
                        roots.append(str(child.GetPath()))
        except Exception:
            roots = []

    for root_path in roots:
        try:
            root_path = str(root_path)
            box_path = root_path.rstrip("/") + "/" + str(ORIBOX_STACK_BOX_MESH_NAME)
            root_prim = stage.GetPrimAtPath(root_path)
            box_prim = stage.GetPrimAtPath(box_path)
            if not (root_prim and root_prim.IsValid() and box_prim and box_prim.IsValid()):
                continue
            root_t = get_world_translation(stage, root_path)
            box_t = get_world_translation(stage, box_path)
            if root_t is None or box_t is None:
                continue
            delta = float(np.linalg.norm(np.array(box_t, dtype=float) - np.array(root_t, dtype=float)))
            if delta > float(ORIBOX_CENTER_MATCH_TOL):
                set_prim_world_translation(stage, box_path, root_t, _sync_oribox_child=False)
                if verbose and bool(ORIBOX_CENTER_MATCH_LOG):
                    print(
                        f"  [ORIBOX_CENTER_SETUP_SYNC] {root_path}: "
                        f"root_t={np.array(root_t)}, child_t_before={np.array(box_t)}, delta={delta:.6f} -> child=root"
                    )
            elif verbose and bool(ORIBOX_CENTER_MATCH_LOG):
                print(f"  [ORIBOX_CENTER_SETUP_OK] {root_path}: root/child world translate delta={delta:.6f}")
        except Exception as exc:
            print(f"  [WARN] ORIBOX_CENTER_SETUP_SYNC 실패: {root_path} / {exc}")


def set_prim_kinematic(stage, prim_path, enabled=True):
    """
    scripted suction 중에는 박스를 kinematic으로 만들어 물리 엔진과 싸우지 않게 한다.
    release 때 False로 돌린다.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    try:
        UsdPhysics.RigidBodyAPI.Apply(prim)
    except Exception:
        pass

    attr = prim.GetAttribute("physics:kinematicEnabled")
    if not attr:
        attr = prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool, custom=False)
    try:
        attr.Set(bool(enabled))
    except Exception:
        pass


def zero_prim_velocity(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    for attr_name in ["physics:velocity", "physics:angularVelocity"]:
        try:
            attr = prim.GetAttribute(attr_name)
            if attr:
                attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        except Exception:
            pass




def _get_world_yaw_z_159(stage, prim_path):
    """prim의 local +X축이 world XY에서 바라보는 yaw(rad)를 반환한다."""
    try:
        prim = stage.GetPrimAtPath(str(prim_path))
        if not prim or not prim.IsValid():
            return None
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        mat = cache.GetLocalToWorldTransform(prim)
        x_axis = mat.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        return float(math.atan2(float(x_axis[1]), float(x_axis[0])))
    except Exception:
        return None


def _set_root_world_yaw_preserve_translation_159(stage, prim_path, yaw_rad):
    """prim root의 world translation은 유지하고, world Z yaw만 지정한다. /World 하위 root용."""
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim or not prim.IsValid():
        print(f"[SLOT_YAW_ALIGN_159][WARN] invalid prim: {prim_path}")
        return False
    try:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        parent = prim.GetParent()
        current_world = cache.GetLocalToWorldTransform(prim)
        world_t = current_world.ExtractTranslation()

        world_mat = Gf.Matrix4d(1.0)
        world_mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), math.degrees(float(yaw_rad))))
        world_mat.SetTranslateOnly(world_t)

        if parent and parent.IsValid():
            parent_world = cache.GetLocalToWorldTransform(parent)
            local_mat = world_mat * parent_world.GetInverse()
        else:
            local_mat = world_mat

        op = _get_or_create_scripted_transform_op(prim)
        op.Set(local_mat)
        return True
    except Exception as exc:
        print(f"[SLOT_YAW_ALIGN_159][WARN] set yaw failed: {type(exc).__name__}: {exc}")
        return False


def get_slot_marker_path_for_slot_159(slot_index):
    paths = tuple(globals().get("SLOT_MARKER_PATHS_155", ()))
    if not paths:
        return None
    idx = int(slot_index) % len(paths)
    return str(paths[idx])


def align_box_yaw_to_slot_marker_159(stage, box_root_path, slot_index):
    """release 직전 박스 root yaw를 APalt_slot marker yaw와 동일하게 맞춘다."""
    if not bool(globals().get("SLOT_MARKER_YAW_ALIGN_ENABLED_159", False)):
        return False
    marker_path = get_slot_marker_path_for_slot_159(slot_index)
    if not marker_path:
        print("[SLOT_YAW_ALIGN_159][WARN] no marker path")
        return False
    target_yaw = _get_world_yaw_z_159(stage, marker_path)
    current_yaw = _get_world_yaw_z_159(stage, box_root_path)
    if target_yaw is None or current_yaw is None:
        print(f"[SLOT_YAW_ALIGN_159][WARN] yaw read failed: box={box_root_path}, marker={marker_path}")
        return False
    ok = _set_root_world_yaw_preserve_translation_159(stage, box_root_path, target_yaw)
    if bool(globals().get("SLOT_MARKER_YAW_ALIGN_LOG_159", True)):
        delta = float((target_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi)
        print(
            f"[SLOT_YAW_ALIGN_159] box={box_root_path}, marker={marker_path}, "
            f"current_yaw={math.degrees(current_yaw):+.1f}deg, target_yaw={math.degrees(target_yaw):+.1f}deg, "
            f"delta={math.degrees(delta):+.1f}deg, ok={ok}"
        )
    return bool(ok)



def check_box_yaw_against_slot_marker_160(stage, box_root_path, slot_index, label="before_release"):
    """160_: release 직전 yaw를 강제로 바꾸지 않고, marker와 차이만 로그로 남긴다."""
    marker_path = get_slot_marker_path_for_slot_159(slot_index)
    if not marker_path:
        print(f"[YAW_CHECK_160][WARN] no marker path. box={box_root_path}, slot_index={slot_index}, label={label}")
        return None
    target_yaw = _get_world_yaw_z_159(stage, marker_path)
    current_yaw = _get_world_yaw_z_159(stage, box_root_path)
    if target_yaw is None or current_yaw is None:
        print(f"[YAW_CHECK_160][WARN] yaw read failed. box={box_root_path}, marker={marker_path}, label={label}")
        return None
    delta = float((target_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi)
    ok = abs(math.degrees(delta)) <= 5.0
    print(
        f"[YAW_CHECK_160] label={label}, box={box_root_path}, marker={marker_path}, "
        f"box_yaw={math.degrees(current_yaw):+.1f}deg, marker_yaw={math.degrees(target_yaw):+.1f}deg, "
        f"yaw_err={math.degrees(delta):+.1f}deg, ok={ok}, snap_disabled=True"
    )
    return delta


def yaw_diagnostic_against_slot_163(stage, box_root_path, slot_index, label="pre_pick"):
    """163_: 상자를 돌리지 않고 현재 box yaw와 APalt_slot yaw 차이만 기록한다."""
    if not bool(globals().get("PRE_PICK_YAW_DIAGNOSTIC_ENABLED_163", True)):
        return None
    marker_path = get_slot_marker_path_for_slot_159(slot_index)
    if not marker_path:
        print(f"[PRE_PICK_YAW_163][WARN] no marker path. box={box_root_path}, slot_index={slot_index}, label={label}")
        return None
    box_yaw = _get_world_yaw_z_159(stage, box_root_path)
    marker_yaw = _get_world_yaw_z_159(stage, marker_path)
    if box_yaw is None or marker_yaw is None:
        print(f"[PRE_PICK_YAW_163][WARN] yaw read failed. box={box_root_path}, marker={marker_path}, label={label}")
        return None
    yaw_err = float((marker_yaw - box_yaw + math.pi) % (2.0 * math.pi) - math.pi)
    yaw_err_deg = float(math.degrees(yaw_err))
    abs_err = abs(yaw_err_deg)
    ok_tol = float(globals().get("PRE_PICK_YAW_OK_TOL_DEG_163", 5.0))
    warn_tol = float(globals().get("PRE_PICK_YAW_WARN_TOL_DEG_163", 15.0))
    status = "OK" if abs_err <= ok_tol else ("WARN" if abs_err <= warn_tol else "BAD")
    print(
        f"[PRE_PICK_YAW_163] label={label}, box={box_root_path}, marker={marker_path}, "
        f"box_yaw={math.degrees(box_yaw):+.1f}deg, marker_yaw={math.degrees(marker_yaw):+.1f}deg, "
        f"yaw_err={yaw_err_deg:+.1f}deg, status={status}, "
        f"post_pick_yaw_correction_disabled={bool(globals().get('POST_PICK_YAW_CORRECTION_DISABLED_163', True))}"
    )
    return yaw_err


def _wrap_deg_164(deg):
    """[-180, 180) 범위로 각도 정규화."""
    try:
        return float((float(deg) + 180.0) % 360.0 - 180.0)
    except Exception:
        return 0.0


def _axis_equiv_err_deg_164(deg):
    """직사각형의 축 방향만 볼 때 180도 동일 방향으로 취급한 최소 오차."""
    d = abs(_wrap_deg_164(deg))
    return float(min(d, abs(180.0 - d)))


def _get_world_axis_yaws_z_164(stage, prim_path):
    """prim local X/Y축이 world XY에서 바라보는 yaw(deg)를 함께 반환한다."""
    try:
        prim = stage.GetPrimAtPath(str(prim_path))
        if not prim or not prim.IsValid():
            return None
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        mat = cache.GetLocalToWorldTransform(prim)
        t = mat.ExtractTranslation()
        x_axis = mat.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        y_axis = mat.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
        z_axis = mat.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
        x_yaw = math.degrees(math.atan2(float(x_axis[1]), float(x_axis[0])))
        y_yaw = math.degrees(math.atan2(float(y_axis[1]), float(y_axis[0])))
        return {
            "path": str(prim_path),
            "pos": (float(t[0]), float(t[1]), float(t[2])),
            "x_yaw": _wrap_deg_164(x_yaw),
            "y_yaw": _wrap_deg_164(y_yaw),
            "x_axis": (float(x_axis[0]), float(x_axis[1]), float(x_axis[2])),
            "y_axis": (float(y_axis[0]), float(y_axis[1]), float(y_axis[2])),
            "z_axis": (float(z_axis[0]), float(z_axis[1]), float(z_axis[2])),
        }
    except Exception:
        return None


def axis_yaw_diagnostic_164(stage, box_root_path, slot_index, label="axis_yaw"):
    """164_: 강제 보정 없이 box와 APalt_slot의 local X/Y축 방향을 모두 진단한다."""
    if not bool(globals().get("AXIS_YAW_DIAGNOSTIC_ENABLED_164", True)):
        return None
    marker_path = get_slot_marker_path_for_slot_159(slot_index)
    if not marker_path:
        print(f"[BOX_AXIS_YAW_164][WARN] no marker path. label={label}, box={box_root_path}, slot_index={slot_index}")
        return None
    b = _get_world_axis_yaws_z_164(stage, box_root_path)
    m = _get_world_axis_yaws_z_164(stage, marker_path)
    if b is None or m is None:
        print(f"[BOX_AXIS_YAW_164][WARN] axis read failed. label={label}, box={box_root_path}, marker={marker_path}")
        return None

    err_xx = _wrap_deg_164(m["x_yaw"] - b["x_yaw"])
    err_xy = _wrap_deg_164(m["x_yaw"] - b["y_yaw"])
    err_yx = _wrap_deg_164(m["y_yaw"] - b["x_yaw"])
    err_yy = _wrap_deg_164(m["y_yaw"] - b["y_yaw"])

    axis_candidates = {
        "boxX_to_slotX": _axis_equiv_err_deg_164(err_xx),
        "boxY_to_slotX": _axis_equiv_err_deg_164(err_xy),
        "boxX_to_slotY": _axis_equiv_err_deg_164(err_yx),
        "boxY_to_slotY": _axis_equiv_err_deg_164(err_yy),
    }
    best_axis = min(axis_candidates, key=axis_candidates.get)
    best_axis_err = float(axis_candidates[best_axis])

    bbox_txt = ""
    if bool(globals().get("AXIS_YAW_DIAG_LOG_BBOX_164", True)):
        try:
            bb = get_world_bbox_info(stage, box_root_path)
            if bb is not None:
                c = bb.get("center")
                s = bb.get("size")
                bbox_txt = (
                    f", bbox_center=({float(c[0]):+.4f},{float(c[1]):+.4f},{float(c[2]):+.4f})"
                    f", bbox_size=({float(s[0]):.4f},{float(s[1]):.4f},{float(s[2]):.4f})"
                )
        except Exception:
            bbox_txt = ""

    print(
        f"[BOX_AXIS_YAW_164] label={label}, slot_index={slot_index}, "
        f"box={box_root_path}, marker={marker_path}, "
        f"box_x_yaw={b['x_yaw']:+.1f}deg, box_y_yaw={b['y_yaw']:+.1f}deg, "
        f"slot_x_yaw={m['x_yaw']:+.1f}deg, slot_y_yaw={m['y_yaw']:+.1f}deg, "
        f"direct_errs: slotX-boxX={err_xx:+.1f}, slotX-boxY={err_xy:+.1f}, "
        f"slotY-boxX={err_yx:+.1f}, slotY-boxY={err_yy:+.1f}, "
        f"axis180_best={best_axis}:{best_axis_err:.1f}deg"
        f"{bbox_txt}"
    )
    print(
        f"[BOX_AXIS_VECTOR_164] label={label}, "
        f"boxX=({b['x_axis'][0]:+.4f},{b['x_axis'][1]:+.4f},{b['x_axis'][2]:+.4f}), "
        f"boxY=({b['y_axis'][0]:+.4f},{b['y_axis'][1]:+.4f},{b['y_axis'][2]:+.4f}), "
        f"slotX=({m['x_axis'][0]:+.4f},{m['x_axis'][1]:+.4f},{m['x_axis'][2]:+.4f}), "
        f"slotY=({m['y_axis'][0]:+.4f},{m['y_axis'][1]:+.4f},{m['y_axis'][2]:+.4f})"
    )
    return {
        "box": b,
        "marker": m,
        "direct_errs": {"slotX-boxX": err_xx, "slotX-boxY": err_xy, "slotY-boxX": err_yx, "slotY-boxY": err_yy},
        "axis180_best": best_axis,
        "axis180_best_err": best_axis_err,
    }



def _vec3_np_170(v):
    try:
        return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float)
    except Exception:
        return np.zeros(3, dtype=float)


def _angle_between_deg_170(a, b):
    """두 3D 벡터의 각도(deg). 진단용이며 어떤 pose도 수정하지 않는다."""
    try:
        av = _vec3_np_170(a)
        bv = _vec3_np_170(b)
        an = float(np.linalg.norm(av))
        bn = float(np.linalg.norm(bv))
        if an < 1.0e-9 or bn < 1.0e-9:
            return None
        c = float(np.clip(np.dot(av, bv) / (an * bn), -1.0, 1.0))
        return float(math.degrees(math.acos(c)))
    except Exception:
        return None


def apalt_release_pose_diagnose_only_170(stage, box_root_path, slot_index, label="before_release"):
    """
    170_: release 직전 진단 전용.
    - 상자/slot transform을 절대 수정하지 않는다.
    - joint/RMPFlow/surface 상태를 절대 변경하지 않는다.
    - APalt_slot을 정답지로 보고 현재 box 중심/수평/축 방향 차이만 출력한다.
    """
    if not bool(globals().get("APALT_RELEASE_POSE_DIAG_ONLY_170", True)):
        return None

    marker_path = get_slot_marker_path_for_slot_159(slot_index)
    if not marker_path:
        print(f"[APALT_RELEASE_POSE_DIAG_170][WARN] no marker path. label={label}, box={box_root_path}, slot_index={slot_index}")
        return None

    b = _get_world_axis_yaws_z_164(stage, box_root_path)
    m = _get_world_axis_yaws_z_164(stage, marker_path)
    if b is None or m is None:
        print(f"[APALT_RELEASE_POSE_DIAG_170][WARN] axis read failed. label={label}, box={box_root_path}, marker={marker_path}")
        return None

    bb = None
    mb = None
    try:
        bb = get_world_bbox_info(stage, box_root_path)
    except Exception:
        bb = None
    try:
        mb = get_world_bbox_info(stage, marker_path)
    except Exception:
        mb = None

    box_center = _vec3_np_170(bb.get("center")) if isinstance(bb, dict) and bb.get("center") is not None else _vec3_np_170(b.get("pos"))
    slot_center = _vec3_np_170(mb.get("center")) if isinstance(mb, dict) and mb.get("center") is not None else _vec3_np_170(m.get("pos"))
    center_xy_err = float(np.linalg.norm(box_center[:2] - slot_center[:2]))
    center_z_err = float(box_center[2] - slot_center[2])

    # 방향 후보는 기존 axis_yaw_diagnostic_164와 같은 방식으로 계산한다.
    err_xx = _wrap_deg_164(m["x_yaw"] - b["x_yaw"])
    err_xy = _wrap_deg_164(m["x_yaw"] - b["y_yaw"])
    err_yx = _wrap_deg_164(m["y_yaw"] - b["x_yaw"])
    err_yy = _wrap_deg_164(m["y_yaw"] - b["y_yaw"])
    axis_candidates = {
        "boxX_to_slotX": _axis_equiv_err_deg_164(err_xx),
        "boxY_to_slotX": _axis_equiv_err_deg_164(err_xy),
        "boxX_to_slotY": _axis_equiv_err_deg_164(err_yx),
        "boxY_to_slotY": _axis_equiv_err_deg_164(err_yy),
    }
    axis_signed_errors = {
        "boxX_to_slotX": float(err_xx),
        "boxY_to_slotX": float(err_xy),
        "boxX_to_slotY": float(err_yx),
        "boxY_to_slotY": float(err_yy),
    }
    best_axis = min(axis_candidates, key=axis_candidates.get)
    best_axis_err = float(axis_candidates[best_axis])
    best_axis_signed_err = float(axis_signed_errors.get(best_axis, 0.0))
    if abs(best_axis_signed_err) > 90.0:
        # 축은 180도 동치로 보므로, 실제 보정 명령도 ±90도 안쪽의 가장 가까운 방향으로 접는다.
        best_axis_signed_err = _wrap_deg_164(best_axis_signed_err + (180.0 if best_axis_signed_err < 0 else -180.0))

    box_z = _vec3_np_170(b.get("z_axis"))
    slot_z = _vec3_np_170(m.get("z_axis"))
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    box_to_slot_z_deg = _angle_between_deg_170(box_z, slot_z)
    box_to_world_up_deg = _angle_between_deg_170(box_z, world_up)
    slot_to_world_up_deg = _angle_between_deg_170(slot_z, world_up)

    # 높이/크기 로그: 기울면 bbox height가 커지는 경향이 있어 같이 기록한다.
    bbox_size_txt = ""
    try:
        if isinstance(bb, dict) and bb.get("size") is not None:
            s = bb.get("size")
            bbox_size_txt = f", box_bbox_size=({float(s[0]):.4f},{float(s[1]):.4f},{float(s[2]):.4f})"
    except Exception:
        bbox_size_txt = ""

    ok_center = center_xy_err <= float(globals().get("APALT_RELEASE_CENTER_OK_TOL_170", 0.010))
    ok_axis = best_axis_err <= float(globals().get("APALT_RELEASE_AXIS_OK_TOL_DEG_170", 5.0))
    ok_level = (box_to_slot_z_deg is not None) and (box_to_slot_z_deg <= float(globals().get("APALT_RELEASE_LEVEL_OK_TOL_DEG_170", 5.0)))

    print(
        f"[APALT_RELEASE_POSE_DIAG_170] label={label}, box={box_root_path}, marker={marker_path}, "
        f"center_xy_err={center_xy_err:.4f}, center_z_err={center_z_err:+.4f}, "
        f"best_axis={best_axis}:{best_axis_err:.1f}deg, signed_yaw_cmd={best_axis_signed_err:+.1f}deg, "
        f"box_to_slot_z={box_to_slot_z_deg if box_to_slot_z_deg is not None else float('nan'):.2f}deg, "
        f"box_to_world_up={box_to_world_up_deg if box_to_world_up_deg is not None else float('nan'):.2f}deg, "
        f"slot_to_world_up={slot_to_world_up_deg if slot_to_world_up_deg is not None else float('nan'):.2f}deg, "
        f"ok_center={ok_center}, ok_axis={ok_axis}, ok_level={ok_level}, "
        f"NO_MOVE=True, NO_RMPFLOW_ALIGN=True, NO_JOINT_TRIM=True, NO_SNAP=True{bbox_size_txt}"
    )
    print(
        f"[APALT_RELEASE_AXIS_CANDIDATES_170] label={label}, "
        f"slotX-boxX={err_xx:+.1f}/{axis_candidates['boxX_to_slotX']:.1f}, "
        f"slotX-boxY={err_xy:+.1f}/{axis_candidates['boxY_to_slotX']:.1f}, "
        f"slotY-boxX={err_yx:+.1f}/{axis_candidates['boxX_to_slotY']:.1f}, "
        f"slotY-boxY={err_yy:+.1f}/{axis_candidates['boxY_to_slotY']:.1f}"
    )
    return {
        "center_xy_err": center_xy_err,
        "center_z_err": center_z_err,
        "best_axis": best_axis,
        "best_axis_err": best_axis_err,
        "best_axis_signed_err": best_axis_signed_err,
        "box_to_slot_z_deg": box_to_slot_z_deg,
        "box_to_world_up_deg": box_to_world_up_deg,
        "slot_to_world_up_deg": slot_to_world_up_deg,
        "ok_center": ok_center,
        "ok_axis": ok_axis,
        "ok_level": ok_level,
    }




def _quat_wxyz_normalize_171(q):
    q = np.array(q, dtype=float).reshape(-1)
    if q.size < 4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q = q[:4].astype(float)
    n = float(np.linalg.norm(q))
    if n < 1.0e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def _quat_wxyz_mul_171(q1, q2):
    """Isaac 관례인 [w,x,y,z] quaternion multiply: q = q1 * q2."""
    w1, x1, y1, z1 = _quat_wxyz_normalize_171(q1)
    w2, x2, y2, z2 = _quat_wxyz_normalize_171(q2)
    return _quat_wxyz_normalize_171(np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float))


def _quat_wxyz_from_world_z_yaw_deg_171(yaw_deg):
    half = math.radians(float(yaw_deg)) * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)


def _make_pre_release_yaw_target_quat_171(robot, yaw_cmd_deg):
    """
    현재 EE orientation에 world-Z yaw 보정량을 pre-multiply한다.
    상자 transform은 건드리지 않고, RMPFlow target orientation으로만 사용한다.
    """
    _ee_pos, ee_quat = get_current_ee_pose(robot)
    q_now = _quat_wxyz_normalize_171(ee_quat)
    q_yaw = _quat_wxyz_from_world_z_yaw_deg_171(float(yaw_cmd_deg))
    return _quat_wxyz_mul_171(q_yaw, q_now)


def _prepare_pre_release_yaw_align_phase_171(stage, robot, task, phase_info, suction_now):
    """
    slot_marker_settle_155 이후 release 직전에 1회만 호출된다.
    현재 box vs APalt_slot 진단값을 보고 yaw만 RMPFlow target orientation으로 보정할지 결정한다.
    """
    if phase_info.get("prepared_171", False):
        return phase_info
    phase_info["prepared_171"] = True
    phase_info["start"] = np.array(suction_now, dtype=float).copy()
    phase_info["target"] = np.array(suction_now, dtype=float).copy()
    phase_info["steps"] = int(globals().get("PRE_RELEASE_YAW_ALIGN_STEPS_171", 70))

    slot_idx = int(getattr(task, "_stack_slot_index", 0))
    diag = apalt_release_pose_diagnose_only_170(stage, task.box_move_path, slot_idx, label="before_pre_release_yaw_align_173")
    phase_info["diag_before_171"] = diag
    if not isinstance(diag, dict):
        phase_info["skip_171"] = True
        phase_info["skip_reason_171"] = "diag_failed"
        phase_info["steps"] = 1
        print("[PRE_RELEASE_YAW_ALIGN_171] skip: diag_failed")
        return phase_info

    center_ok = bool(diag.get("ok_center", False))
    level_ok = bool(diag.get("ok_level", False))
    axis_ok = bool(diag.get("ok_axis", False))
    yaw_cmd = float(diag.get("best_axis_signed_err", 0.0))
    yaw_abs = abs(yaw_cmd)
    min_deg = float(globals().get("PRE_RELEASE_YAW_ALIGN_MIN_DEG_171", 2.0))
    max_deg = float(globals().get("PRE_RELEASE_YAW_ALIGN_MAX_DEG_171", 25.0))

    if bool(globals().get("PRE_RELEASE_YAW_ALIGN_ONLY_IF_CENTER_LEVEL_OK_171", True)) and (not center_ok or not level_ok):
        phase_info["skip_171"] = True
        phase_info["skip_reason_171"] = f"center_or_level_not_ok:center={center_ok},level={level_ok}"
        phase_info["steps"] = 1
    elif axis_ok or yaw_abs < min_deg:
        phase_info["skip_171"] = True
        phase_info["skip_reason_171"] = f"axis_already_ok_or_small:yaw={yaw_cmd:+.2f}"
        phase_info["steps"] = 1
    elif yaw_abs > max_deg:
        phase_info["skip_171"] = True
        phase_info["skip_reason_171"] = f"yaw_too_large:{yaw_cmd:+.2f}>{max_deg:.1f}"
        phase_info["steps"] = 1
    else:
        # yaw_cmd는 slot yaw - box yaw 계열이므로, EE 목표에도 같은 world-Z yaw 보정량을 준다.
        phase_info["skip_171"] = False
        phase_info["yaw_cmd_deg_171"] = yaw_cmd
        phase_info["fixed_orientation_171"] = _make_pre_release_yaw_target_quat_171(robot, yaw_cmd)
        phase_info["hold_suction_171"] = np.array(suction_now, dtype=float).copy()
        phase_info["target"] = np.array(suction_now, dtype=float).copy()
        print(
            f"[PRE_RELEASE_YAW_ALIGN_173] start. slot={slot_idx}, yaw_cmd={yaw_cmd:+.2f}deg, "
            f"best_axis={diag.get('best_axis')}:{float(diag.get('best_axis_err', 999.0)):.2f}deg, "
            f"center_xy_err={float(diag.get('center_xy_err', 999.0)):.4f}, "
            f"level={float(diag.get('box_to_slot_z_deg', 999.0)):.2f}deg, "
            f"steps={phase_info['steps']}, mode=REAL_RMPFLOW_TARGET_ORIENTATION_172_METHOD, hold_suction=True, full_steps=True, no_box_transform=True, no_joint6_trim=True"
        )

    if bool(phase_info.get("skip_171", False)):
        print(f"[PRE_RELEASE_YAW_ALIGN_171] skip_reason={phase_info.get('skip_reason_171')}")
    return phase_info



def _is_fused_yaw_slot_184(task):
    try:
        slot_idx = int(getattr(task, "_stack_slot_index", 0))
        allowed = tuple(int(x) for x in globals().get("FUSED_YAW_ALIGN_SLOT_INDICES_184", (1,)))
        return bool(slot_idx in allowed)
    except Exception:
        return False


def _prepare_fused_yaw_align_phase_184(stage, robot, task, phase_info, suction_now):
    """
    184_: release 직전 별도 pre_release_yaw_align_171 phase를 기다리지 않고,
    slot_marker_lower_155/settle phase의 RMPFlow target orientation에 yaw 보정을 섞는다.
    위치 목표(start/target)는 절대 바꾸지 않는다. 즉 내려가면서 yaw만 같이 맞춘다.
    """
    try:
        if not bool(globals().get("FUSED_SECOND_BOX_YAW_ALIGN_ENABLED_184", False)):
            return phase_info
        if not isinstance(phase_info, dict):
            return phase_info
        phase_name = str(phase_info.get("name", ""))
        if phase_name not in tuple(globals().get("FUSED_YAW_ALIGN_PHASE_NAMES_184", ("slot_marker_lower_155", "slot_marker_settle_155"))):
            return phase_info
        if not _is_fused_yaw_slot_184(task):
            return phase_info

        # settle에서는 lower에서 계산한 orientation을 재사용한다.
        if phase_name != str(globals().get("FUSED_YAW_ALIGN_START_PHASE_184", "slot_marker_lower_155")):
            if bool(globals().get("FUSED_YAW_ALIGN_REUSE_IN_SETTLE_184", True)):
                q_reuse = getattr(task, "_fused_yaw_orientation_184", None)
                if q_reuse is not None:
                    phase_info["fixed_orientation_171"] = np.array(q_reuse, dtype=float).copy()
                    phase_info["fused_yaw_reused_184"] = True
                    if bool(globals().get("FUSED_YAW_ALIGN_LOG_184", True)) and not phase_info.get("fused_yaw_reuse_logged_184", False):
                        phase_info["fused_yaw_reuse_logged_184"] = True
                        print(f"[FUSED_YAW_ALIGN_184] reuse orientation during {phase_name}; separate pre_release_yaw_align skipped")
            return phase_info

        if phase_info.get("fused_yaw_prepared_184", False):
            return phase_info
        phase_info["fused_yaw_prepared_184"] = True

        slot_idx = int(getattr(task, "_stack_slot_index", 0))
        diag = apalt_release_pose_diagnose_only_170(stage, task.box_move_path, slot_idx, label=f"before_fused_yaw_align_184:{phase_name}")
        phase_info["fused_yaw_diag_184"] = diag
        if not isinstance(diag, dict):
            phase_info["fused_yaw_skip_184"] = "diag_failed"
            if bool(globals().get("FUSED_YAW_ALIGN_LOG_184", True)):
                print(f"[FUSED_YAW_ALIGN_184] skip: diag_failed during {phase_name}")
            return phase_info

        center_xy = float(diag.get("center_xy_err", 999.0))
        level = float(diag.get("box_to_slot_z_deg", 999.0))
        axis_ok = bool(diag.get("ok_axis", False))
        yaw_cmd = float(diag.get("best_axis_signed_err", 0.0))
        yaw_abs = abs(yaw_cmd)
        min_deg = float(globals().get("FUSED_YAW_ALIGN_MIN_DEG_184", 2.0))
        max_deg = float(globals().get("FUSED_YAW_ALIGN_MAX_DEG_184", 25.0))
        center_tol = float(globals().get("FUSED_YAW_ALIGN_CENTER_TOL_M_184", 0.080))
        level_tol = float(globals().get("FUSED_YAW_ALIGN_LEVEL_TOL_DEG_184", 5.0))

        if axis_ok or yaw_abs < min_deg:
            phase_info["fused_yaw_skip_184"] = f"axis_already_ok_or_small:yaw={yaw_cmd:+.2f}"
        elif yaw_abs > max_deg:
            phase_info["fused_yaw_skip_184"] = f"yaw_too_large:{yaw_cmd:+.2f}>{max_deg:.1f}"
        elif center_xy > center_tol:
            phase_info["fused_yaw_skip_184"] = f"center_too_far:{center_xy:.4f}>{center_tol:.4f}"
        elif level > level_tol:
            phase_info["fused_yaw_skip_184"] = f"level_not_ok:{level:.2f}>{level_tol:.2f}"
        else:
            q = _make_pre_release_yaw_target_quat_171(robot, yaw_cmd)
            phase_info["fixed_orientation_171"] = q.copy()
            phase_info["fused_yaw_cmd_deg_184"] = yaw_cmd
            phase_info["fused_yaw_active_184"] = True
            setattr(task, "_fused_yaw_orientation_184", q.copy())
            setattr(task, "_fused_yaw_cmd_deg_184", yaw_cmd)
            if bool(globals().get("FUSED_YAW_ALIGN_LOG_184", True)):
                print(
                    f"[FUSED_YAW_ALIGN_184] active during {phase_name}. slot={slot_idx}, "
                    f"yaw_cmd={yaw_cmd:+.2f}deg, center_xy_err={center_xy:.4f}, level={level:.2f}deg, "
                    f"separate_pre_release_phase={not bool(globals().get('FUSED_YAW_ALIGN_DISABLE_FINAL_PRE_RELEASE_PHASE_184', True))}, "
                    f"mode=LOWER_AND_SETTLE_RMPFLOW_TARGET_ORIENTATION"
                )
            return phase_info

        if bool(globals().get("FUSED_YAW_ALIGN_LOG_184", True)):
            print(f"[FUSED_YAW_ALIGN_184] skip during {phase_name}: {phase_info.get('fused_yaw_skip_184')}")
        return phase_info
    except Exception as exc:
        try:
            print(f"[FUSED_YAW_ALIGN_184][WARN] prepare failed: {type(exc).__name__}: {exc}")
        except Exception:
            pass
        return phase_info

def _diag_stop_after_release_count_reached_164(release_count):
    try:
        return bool(globals().get("DIAG_STOP_AFTER_RELEASE_ENABLED_164", True)) and int(release_count) >= int(globals().get("DIAG_STOP_AFTER_RELEASE_COUNT_164", 2))
    except Exception:
        return False


def zero_subtree_velocity(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return 0
    count = 0
    for prim in Usd.PrimRange(root):
        for attr_name in ["physics:velocity", "physics:angularVelocity"]:
            try:
                attr = prim.GetAttribute(attr_name)
                if attr:
                    attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    count += 1
            except Exception:
                pass
    return count


def set_box_scripted_carry_mode(stage, root_path, enabled=True, reenable_physics=False, verbose=True):
    """
    15_ 핵심.
    VGC10은 visual-only라 물리 충돌로 박스를 밀 수 없다.
    scripted suction으로 박스를 들어올릴 때는 박스 subtree의 물리 rigid/collision을 잠시 끄고
    parent Xform을 직접 움직여야 허우적거림과 관통 느낌이 줄어든다.

    enabled=True  : carry 중. rigidBodyEnabled=False, collisionEnabled=False, kinematic=True, velocity=0
    enabled=False : release 후. 기본은 reenable_physics=False라 목표 위치에 안정적으로 고정한다.
                    dynamic으로 되돌리고 싶으면 BOX_REENABLE_PHYSICS_AFTER_RELEASE=True.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        if verbose:
            print(f"  [CARRY_MODE_FAIL] root not found: {root_path}")
        return {"rigid": 0, "collision": 0, "kinematic": 0, "velocity": 0}

    stats = {"rigid": 0, "collision": 0, "kinematic": 0, "velocity": 0}
    for prim in Usd.PrimRange(root):
        # 속도 제거
        for attr_name in ["physics:velocity", "physics:angularVelocity"]:
            try:
                attr = prim.GetAttribute(attr_name)
                if attr:
                    attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    stats["velocity"] += 1
            except Exception:
                pass

        # rigid body 제어
        try:
            has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAttribute("physics:rigidBodyEnabled") or prim.HasAttribute("physics:kinematicEnabled")
            if has_rb:
                UsdPhysics.RigidBodyAPI.Apply(prim)
                kin_attr = prim.GetAttribute("physics:kinematicEnabled")
                if not kin_attr:
                    kin_attr = prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool, custom=False)
                kin_attr.Set(True if enabled else (not reenable_physics))
                stats["kinematic"] += 1

                rb_attr = prim.GetAttribute("physics:rigidBodyEnabled")
                if rb_attr:
                    rb_attr.Set(False if enabled else bool(reenable_physics))
                    stats["rigid"] += 1
        except Exception:
            pass

        # collision 제어
        try:
            has_col = prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAttribute("physics:collisionEnabled")
            if has_col:
                col_api = UsdPhysics.CollisionAPI.Apply(prim)
                col_attr = prim.GetAttribute("physics:collisionEnabled")
                if not col_attr:
                    col_attr = col_api.CreateCollisionEnabledAttr()
                col_attr.Set(False if enabled else bool(reenable_physics))
                stats["collision"] += 1
        except Exception:
            pass

    if verbose:
        print(f"  [CARRY_MODE] enabled={enabled}, reenable_physics={reenable_physics}, root={root_path}, stats={stats}")
    return stats


def _vec_mag(value):
    if value is None:
        return None
    try:
        arr = np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)
        return float(np.linalg.norm(arr))
    except Exception:
        return None


def get_prim_velocity_magnitudes(stage, prim_path):
    """
    USD/PhysX 속성에서 linear/angular velocity 크기를 읽는다.
    런타임에서 attr이 비어 있으면 None을 반환하고, 이 경우 bbox 이동량으로 정지 여부를 판단한다.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None, None

    linear_mag = None
    angular_mag = None
    try:
        attr = prim.GetAttribute("physics:velocity")
        if attr:
            linear_mag = _vec_mag(attr.Get())
    except Exception:
        linear_mag = None

    try:
        attr = prim.GetAttribute("physics:angularVelocity")
        if attr:
            angular_mag = _vec_mag(attr.Get())
    except Exception:
        angular_mag = None

    return linear_mag, angular_mag


class BoxStopDetector:
    """
    Small_Cardboard_box 하나가 멈췄는지 판정한다.

    8_ 수정 핵심:
    - gate 기준은 bbox center 이동량이다.
    - physics velocity는 Isaac/PhysX attr 잔류값이 남을 수 있어서 기본값으로는 gate에 쓰지 않는다.
    - 대신 매 step pos/linear/angular 각각 PASS/FAIL/IGNORED를 로그로 찍어서 왜 대기 중인지 바로 보이게 한다.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_center = None
        self.stable_steps = 0
        self.total_steps = 0
        self.last_move = 999.0
        self.last_linear_vel = None
        self.last_angular_vel = None
        self.stable_center = None
        self.last_blockers = []

    @staticmethod
    def _fmt_gate(name, enabled, value, tol, ok, unit=""):
        if value is None:
            val_s = "None"
        elif abs(value) >= 100.0 or abs(value) < 0.0001:
            val_s = f"{value:.6f}"
        else:
            val_s = f"{value:.5f}"

        if not enabled:
            state = "INFO_ONLY"
        else:
            state = "OK" if ok else "FAIL"
        return f"{name}={val_s}{unit}<={tol:.5f}{unit}:{state}"

    def update(self, stage, prim_path, bbox_info):
        self.total_steps += 1

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            self.prev_center = None
            self.stable_steps = 0
            self.last_move = 999.0
            # 비슷한 이름을 찾아서 로그에 띄운다. 경로 오타를 바로 확인하기 위함.
            matches = []
            try:
                for p in Usd.PrimRange(stage.GetPseudoRoot()):
                    if "small_cardboard_box" in p.GetName().lower() or "small_cardboard_box" in str(p.GetPath()).lower():
                        matches.append(str(p.GetPath()))
            except Exception:
                pass
            return False, f"target_prim_invalid path={prim_path} matches={matches[:8]}"

        if bbox_info is None:
            self.prev_center = None
            self.stable_steps = 0
            self.last_move = 999.0
            return False, f"no_bbox path={prim_path} prim_valid=True"

        center = np.array(bbox_info["center"], dtype=float)
        if self.prev_center is None:
            move = 999.0
        else:
            move = float(np.linalg.norm(center - self.prev_center))

        linear_vel, angular_vel = get_prim_velocity_magnitudes(stage, prim_path)

        pos_raw_ok = move <= BOX_STABLE_POS_TOL
        linear_raw_ok = (linear_vel is None) or (linear_vel <= BOX_STABLE_LINEAR_VEL_TOL)
        angular_raw_ok = (angular_vel is None) or (angular_vel <= BOX_STABLE_ANGULAR_VEL_TOL)

        gate_parts = []
        if BOX_STOP_USE_BBOX_MOVE:
            gate_parts.append(pos_raw_ok)
        if BOX_STOP_USE_LINEAR_VEL:
            gate_parts.append(linear_raw_ok)
        if BOX_STOP_USE_ANGULAR_VEL:
            gate_parts.append(angular_raw_ok)

        # 혹시 모든 gate를 꺼도 bbox 기준으로는 최소 판정한다.
        candidate_ok = all(gate_parts) if gate_parts else pos_raw_ok

        blockers = []
        if BOX_STOP_USE_BBOX_MOVE and not pos_raw_ok:
            blockers.append("MOVE")
        if BOX_STOP_USE_LINEAR_VEL and not linear_raw_ok:
            blockers.append("LINEAR_VEL")
        if BOX_STOP_USE_ANGULAR_VEL and not angular_raw_ok:
            blockers.append("ANGULAR_VEL")

        if candidate_ok:
            self.stable_steps += 1
        else:
            self.stable_steps = 0

        self.prev_center = center.copy()
        self.last_move = move
        self.last_linear_vel = linear_vel
        self.last_angular_vel = angular_vel
        self.last_blockers = blockers

        stopped = self.stable_steps >= BOX_STABLE_REQUIRED_STEPS
        if stopped:
            self.stable_center = center.copy()

        if stopped:
            state = "STOP_CONFIRMED"
        elif candidate_ok:
            state = "COUNTING_STABLE"
        else:
            state = "WAITING_" + ("+".join(blockers) if blockers else "UNKNOWN")

        reason = (
            f"mode=bbox_only_selected_prim,"
            f"state={state},"
            f"stable={self.stable_steps}/{BOX_STABLE_REQUIRED_STEPS},"
            f"step={self.total_steps},"
            f"{self._fmt_gate('move', BOX_STOP_USE_BBOX_MOVE, move, BOX_STABLE_POS_TOL, pos_raw_ok, 'm')},"
            f"{self._fmt_gate('lin_vel', BOX_STOP_USE_LINEAR_VEL, linear_vel, BOX_STABLE_LINEAR_VEL_TOL, linear_raw_ok, 'm/s')},"
            f"{self._fmt_gate('ang_vel', BOX_STOP_USE_ANGULAR_VEL, angular_vel, BOX_STABLE_ANGULAR_VEL_TOL, angular_raw_ok, 'rad/s')},"
            f"blockers={blockers if blockers else 'NONE'}"
        )
        return stopped, reason


def hold_robot_current_pose(robot):
    """박스가 멈출 때까지 로봇 관절을 현재 위치에 유지한다."""
    try:
        current = np.array(robot.get_joint_positions(), dtype=float)
        robot.apply_action(ArticulationAction(joint_positions=current))
    except Exception:
        pass


def is_box_ready_for_pick_zone(bbox_info):
    """
    18_ 추가: 컨베이어 시작 위치/공중 스폰 상태에서 bbox가 잠깐 멈춰 보여도
    pick을 시작하지 않도록 실제 픽업 구간인지 확인한다.
    """
    if not BOX_READY_ZONE_GATE:
        return True, "ready_gate_disabled"
    if bbox_info is None:
        return False, "no_bbox"
    c = np.array(bbox_info["center"], dtype=float)
    reasons = []
    if float(c[1]) > float(BOX_READY_MIN_Y):
        reasons.append(f"Y_NOT_REACHED({c[1]:.3f}>{BOX_READY_MIN_Y:.3f})")
    if float(c[2]) > float(BOX_READY_MAX_CENTER_Z):
        reasons.append(f"Z_TOO_HIGH({c[2]:.4f}>{BOX_READY_MAX_CENTER_Z:.4f})")
    return (len(reasons) == 0), ("READY" if not reasons else "|".join(reasons))



def create_pick_zone_visual(stage):
    """
    1번 로봇 앞 pick 준비 영역을 색 있는 큐브로 표시한다.
    이 prim은 visual marker일 뿐이며 rigid body/collider를 만들지 않는다.
    """
    if not PICK_ZONE_VISUAL_ENABLED:
        # 77_: USD 직접 편집 상태를 보존한다. 기존 prim도 지우지 않고, 새 visual도 만들지 않는다.
        return False

    # 기존 영역 표시가 있으면 지우고 다시 만든다. 위치/크기 변경 시 확실히 반영된다.
    clear_prim_if_exists(stage, PICK_ZONE_PATH)

    cube = UsdGeom.Cube.Define(stage, PICK_ZONE_PATH)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()

    # center/size를 바로 xform으로 적용한다.
    _set_xform_common(
        prim,
        translate=PICK_ZONE_CENTER,
        rotate=(0.0, 0.0, 0.0),
        scale=PICK_ZONE_SIZE,
    )

    # 색/투명도 설정. 물리 API는 적용하지 않는다.
    try:
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr([Gf.Vec3f(float(PICK_ZONE_COLOR[0]), float(PICK_ZONE_COLOR[1]), float(PICK_ZONE_COLOR[2]))])
        gprim.CreateDisplayOpacityAttr([float(PICK_ZONE_OPACITY)])
    except Exception:
        pass

    # 혹시 이전 실행/편집으로 physics/collision API가 남아 있으면 제거 또는 비활성화한다.
    try:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    except Exception:
        pass
    try:
        if prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)
    except Exception:
        pass
    try:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
    except Exception:
        pass

    print(
        f"  [PICK_ZONE_VISUAL] path={PICK_ZONE_PATH}, "
        f"center=({PICK_ZONE_CENTER[0]:.3f},{PICK_ZONE_CENTER[1]:.3f},{PICK_ZONE_CENTER[2]:.3f}), "
        f"size=({PICK_ZONE_SIZE[0]:.3f},{PICK_ZONE_SIZE[1]:.3f},{PICK_ZONE_SIZE[2]:.3f}), "
        f"opacity={PICK_ZONE_OPACITY:.2f}, physics=OFF"
    )
    return True


def is_bbox_center_inside_pick_zone(bbox_info):
    """박스 bbox center가 1번 로봇 앞 감지 영역 안에 들어왔는지 판단한다."""
    if bbox_info is None:
        return False, "no_bbox"

    c = np.array(bbox_info["center"], dtype=float)
    mn = PICK_ZONE_CENTER - PICK_ZONE_SIZE * 0.5
    mx = PICK_ZONE_CENTER + PICK_ZONE_SIZE * 0.5

    inside_xyz = (c >= mn) & (c <= mx)
    inside = bool(np.all(inside_xyz))

    blockers = []
    axes = ["X", "Y", "Z"]
    for i, axis in enumerate(axes):
        if c[i] < mn[i]:
            blockers.append(f"{axis}_LOW({c[i]:.3f}<{mn[i]:.3f})")
        elif c[i] > mx[i]:
            blockers.append(f"{axis}_HIGH({c[i]:.3f}>{mx[i]:.3f})")

    reason = (
        f"center=({c[0]:.3f},{c[1]:.3f},{c[2]:.4f}),"
        f"zone_min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}),"
        f"zone_max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f}),"
        f"inside={inside},blockers={blockers if blockers else 'NONE'}"
    )
    return inside, reason


class PickZoneDetector:
    """
    1번 로봇 앞 영역 진입 감지기.
    정지 판정이 아니라, 박스 중심이 지정한 AABB 영역 안에 들어온 것을 기준으로 pick을 시작한다.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_steps = 0
        self.inside_steps = 0
        self.last_reason = "not_started"

    def update(self, bbox_info):
        self.total_steps += 1
        inside, reason = is_bbox_center_inside_pick_zone(bbox_info)
        if inside:
            self.inside_steps += 1
        else:
            self.inside_steps = 0
        ready = self.inside_steps >= int(PICK_ZONE_REQUIRED_STEPS)
        state = "ZONE_READY" if ready else ("COUNTING_ZONE" if inside else "WAITING_ZONE")
        self.last_reason = (
            f"mode=front_zone,state={state},inside={self.inside_steps}/{PICK_ZONE_REQUIRED_STEPS},"
            f"step={self.total_steps},{reason}"
        )
        return ready, self.last_reason


def find_named_child_or_descendant(stage, root_path: str, name: str):
    """root_path 하위에서 지정 child prim path를 찾는다.

    71_: 사용자가 말한 경로는 Small_cardboard_box이고, 기존 USD들은 Small_Cardboard_box를 쓰기도 했다.
    그래서 name 하나만 보지 않고 ORIBOX_STACK_BOX_MESH_NAME_CANDIDATES를 함께 확인한다.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    names = []
    for n in [name] + list(globals().get("ORIBOX_STACK_BOX_MESH_NAME_CANDIDATES", ())):
        n = str(n)
        if n and n not in names:
            names.append(n)
    for prim in Usd.PrimRange(root):
        if prim.GetName() in names:
            return str(prim.GetPath())
    return None


def discover_oriboxa_stack_candidates(stage, completed_roots=None):
    """
    63_ 구조 대응:
    - 사용자가 OriBoxA_move wrapper Xform을 제거하고 /World/OriBoxA_01 형태로 저장한 USD를 우선 지원한다.
    - ORIBOX_STACK_PARENT_PATH는 기본 /World.
    - 혹시 예전 USD를 열었을 때도 망하지 않도록 /World/OriBoxA_move 하위도 fallback으로 추가 검색한다.
    - 각 root 하위의 Small_Cardboard_box bbox를 계산한다.
    """
    completed_roots = set() if completed_roots is None else set(completed_roots)
    results = []
    seen_roots = set()

    parent_paths = []
    for p in [ORIBOX_STACK_PARENT_PATH, "/World", "/World/OriBoxA_move"]:
        if p and p not in parent_paths:
            parent_paths.append(p)

    for parent_path in parent_paths:
        parent = stage.GetPrimAtPath(parent_path)
        if not parent.IsValid():
            continue

        for child in parent.GetChildren():
            root_name = child.GetName()
            root_path = str(child.GetPath())
            if not root_name.startswith(ORIBOX_STACK_NAME_PREFIXES):
                continue
            if root_path in seen_roots:
                continue
            seen_roots.add(root_path)
            if STACK_SKIP_COMPLETED_BOXES and root_path in completed_roots:
                continue

            # 72_: USD에서 root가 Rigid Body 대표 좌표가 되도록 정리했으므로
            # 실제 후보 bbox/좌표/이동 path는 child가 아니라 root_path를 사용한다.
            box_path = root_path
            bbox = get_world_bbox_info(stage, root_path)
            if bbox is None:
                print(f"  [STACK_SCAN_SKIP] {root_path}: root bbox 계산 실패")
                continue

            results.append({
                "root_name": root_name,
                "root_path": root_path,
                "box_path": box_path,
                "bbox": bbox,
            })

    # 이름 순서 OriBoxA_01, OriBoxA_02, OriBoxA_03 순서 우선
    results.sort(key=lambda x: x["root_name"])
    return results

def select_oriboxa_candidate_in_front_zone(stage, completed_roots, zone_counts_by_root):
    """
    작업영역 안에 들어온 OriBoxA_* 후보를 고른다.
    같은 root가 PICK_ZONE_REQUIRED_STEPS 만큼 연속으로 영역에 있어야 ready=True.
    """
    candidates = discover_oriboxa_stack_candidates(stage, completed_roots=completed_roots)
    active_roots = set()
    best_inside = None
    best_reason = "no_candidate_inside"

    for cand in candidates:
        root_path = cand["root_path"]
        active_roots.add(root_path)
        inside, reason = is_bbox_center_inside_pick_zone(cand["bbox"])
        if inside:
            zone_counts_by_root[root_path] = int(zone_counts_by_root.get(root_path, 0)) + 1
            if best_inside is None:
                best_inside = cand
                best_reason = reason
        else:
            zone_counts_by_root[root_path] = 0

    # 사라졌거나 완료된 root의 카운터 정리
    for old_root in list(zone_counts_by_root.keys()):
        if old_root not in active_roots:
            zone_counts_by_root.pop(old_root, None)

    if best_inside is None:
        return None, False, (
            f"mode=multi_oriboxa_front_zone,state=WAITING_ZONE,"
            f"candidates={len(candidates)},completed={len(completed_roots)},reason={best_reason}"
        )

    root_path = best_inside["root_path"]
    count = int(zone_counts_by_root.get(root_path, 0))
    ready = count >= int(PICK_ZONE_REQUIRED_STEPS)
    state = "ZONE_READY" if ready else "COUNTING_ZONE"
    bbox = best_inside["bbox"]
    c = bbox["center"]
    t = bbox["top_center"]
    reason = (
        f"mode=multi_oriboxa_front_zone,state={state},"
        f"target={root_path},box={best_inside['box_path']},"
        f"inside={count}/{PICK_ZONE_REQUIRED_STEPS},"
        f"center=({c[0]:.3f},{c[1]:.3f},{c[2]:.4f}),"
        f"top=({t[0]:.3f},{t[1]:.3f},{t[2]:.4f}),"
        f"{best_reason}"
    )
    return best_inside, ready, reason


def compute_stack_slot_step_from_bbox(box_bbox):
    """박스 크기를 기준으로 2열 slot 간격을 계산한다."""
    if not STACK_USE_BOX_SIZE_FOR_SLOT_STEP or box_bbox is None:
        return np.array(STACK_MANUAL_SLOT_STEP, dtype=float)
    size = np.array(box_bbox.get("size", np.array([0.30, 0.25, 0.24])), dtype=float)
    if str(STACK_SLOT_AXIS).upper() == "Y":
        return np.array([0.0, float(size[1]) + float(STACK_SLOT_GAP), 0.0], dtype=float)
    return np.array([float(size[0]) + float(STACK_SLOT_GAP), 0.0, 0.0], dtype=float)


def get_stack_slot_col_layer(slot_index):
    """slot index를 2열 x 2층 좌표로 변환한다."""
    columns = max(1, int(STACK_COLUMNS))
    slot_index = int(max(0, min(int(slot_index), int(STACK_SLOT_COUNT) - 1)))
    col = slot_index % columns
    layer = slot_index // columns
    return col, layer


def compute_stack_goal_center(stage, box_bbox, slot_index):
    """
    75_ 팔레타이징 goal 계산.

    기준:
    - APalt는 /World/APalt Xform marker.
    - APalt는 팔레트 윗면 중심 좌표라고 가정한다.
    - 2x1 테스트: slot 1, slot 2를 APalt 중심 기준으로 좌우에 배치한다.
    - 상자 이동/후보/완료 기준은 child가 아니라 OriBoxA_01/OriBoxB_01 같은 root 좌표다.
    """
    if box_bbox is None:
        raise RuntimeError("box bbox가 없어 pallet goal을 계산할 수 없습니다.")

    box_size = np.array(box_bbox.get("size", np.array([0.30, 0.25, 0.24])), dtype=float)
    box_height = max(float(box_size[2]), 0.05)
    step = compute_stack_slot_step_from_bbox(box_bbox)
    step_len = float(abs(step[0]))
    if step_len < 1e-6:
        step_len = float(max(box_size[0], box_size[1]) + STACK_SLOT_GAP)

    robot_origin, robot_right, robot_forward, robot_up = get_robot_frame_axes_for_stack(stage)

    use_ori_marker = bool(globals().get("USE_PALLET_ORI_A_MARKER", True))
    ori_path = str(globals().get("PALLET_ORI_A_PATH", "/World/APalt"))
    support_path = ori_path
    support_center = None
    support_top_z = None

    if use_ori_marker:
        ori_pos = get_world_translation(stage, ori_path)
        if ori_pos is None:
            raise RuntimeError(
                f"APalt 기준 prim을 찾지 못했습니다: {ori_path}\n"
                "USD에서 /World/APalt 를 팔레트 윗면 중심에 만든 뒤 저장하세요."
            )
        support_center = np.array(ori_pos, dtype=float)
        support_top_z = float(support_center[2])
    else:
        # fallback: 예전 Cube/BoxAprop bbox 기준
        support_path = resolve_stack_support_path(stage)
        support_bbox = get_world_bbox_info(stage, support_path)
        if support_bbox is None:
            raise RuntimeError(
                f"적재 받침을 찾지 못했습니다: {STACK_SUPPORT_CUBE_PATH} 또는 후보 이름 {STACK_SUPPORT_NAME_CANDIDATES}"
            )
        support_center = np.array(support_bbox["center"], dtype=float)
        support_top_z = float(support_bbox["max"][2])

    robot_offset = np.array(BOXAPROP_SLOT_OFFSET_ROBOT, dtype=float)
    world_offset = (
        robot_right * float(robot_offset[0])
        + robot_forward * float(robot_offset[1])
        + robot_up * float(robot_offset[2])
        + np.array(STACK_FIRST_SLOT_OFFSET, dtype=float)
    )

    slot_index = int(max(0, min(int(slot_index), int(STACK_SLOT_COUNT) - 1)))
    col, layer = get_stack_slot_col_layer(slot_index)
    columns = max(1, int(STACK_COLUMNS))

    axis_name = str(globals().get("PALLET_SLOT_AXIS", STACK_SLOT_AXIS)).upper()
    slot_axis_vec = robot_forward if axis_name in ("ROBOT_Y", "Y", "FORWARD") else robot_right

    def _slot_goal(idx):
        c, l = get_stack_slot_col_layer(idx)
        centered_col = float(c) - (float(columns) - 1.0) * 0.5
        g = support_center.copy() + world_offset
        # 2x1: APalt를 중심으로 slot 1/2를 좌우에 배치한다.
        g[:3] += slot_axis_vec * centered_col * step_len
        # APalt는 팔레트 윗면 중심이므로, 박스 root/center는 박스 높이 절반만큼 위에 둔다.
        g[2] = support_top_z + (float(l) + 0.5) * box_height + float(STACK_PLACE_Z_CLEARANCE)
        return g

    goal = _slot_goal(slot_index)

    print(
        f"  [APALT_SUPPORT_117] path={support_path}, center/top=({support_center[0]:.3f},{support_center[1]:.3f},{support_center[2]:.3f}), "
        f"top_z={support_top_z:.4f}, axis={axis_name}, columns={STACK_COLUMNS}, layers={STACK_LAYERS}, "
        f"box_size=({box_size[0]:.3f},{box_size[1]:.3f},{box_size[2]:.3f}), step_len={step_len:.3f}"
    )
    for i in range(int(STACK_SLOT_COUNT)):
        sg = _slot_goal(i)
        rr = robot_relative_vector(stage, sg)
        print(
            f"  [APALT_SLOT_TRANSLATE_117] slot={i + 1}/{STACK_SLOT_COUNT}, "
            f"world_translate=({sg[0]:.4f}, {sg[1]:.4f}, {sg[2]:.4f}), "
            f"robot_relative=({rr[0]:+.4f}, {rr[1]:+.4f}, {rr[2]:+.4f})"
        )

    rr_goal = robot_relative_vector(stage, goal)
    print(
        f"  [STACK_GOAL] slot={slot_index + 1}/{STACK_SLOT_COUNT} "
        f"(col={col + 1}/{STACK_COLUMNS}, layer={layer + 1}/{STACK_LAYERS}), "
        f"pallet_marker={support_path}, goal_center=({goal[0]:.4f}, {goal[1]:.4f}, {goal[2]:.4f}), "
        f"robot_relative=({rr_goal[0]:+.4f}, {rr_goal[1]:+.4f}, {rr_goal[2]:+.4f})"
    )
    return goal


def resolve_slot_marker_center_155(stage, slot_index, fallback_center=None):
    """APalt 위 정답지 큐브 bbox center를 읽어서 최종 place center로 사용한다."""
    if not bool(globals().get("SLOT_MARKER_PALLETIZING_ENABLED_155", False)):
        return None, "disabled"
    paths = tuple(str(p) for p in globals().get("SLOT_MARKER_PATHS_155", ()))
    if not paths:
        return None, "no marker paths configured"
    idx = int(max(0, slot_index))
    if idx >= len(paths):
        if bool(globals().get("SLOT_MARKER_WRAP_IF_SHORT_155", True)):
            idx = idx % len(paths)
        else:
            return None, f"slot_index={slot_index} out of marker range len={len(paths)}"
    marker_path = paths[idx]
    bbox = get_world_bbox_info(stage, marker_path)
    if bbox is None:
        return None, f"marker bbox not found: {marker_path}"
    center = np.array(bbox["center"], dtype=float)
    mn = np.array(bbox["min"], dtype=float)
    mx = np.array(bbox["max"], dtype=float)
    size = mx - mn
    print(
        f"  [SLOT_MARKER_155] slot_request={int(slot_index)+1}, marker={marker_path}, "
        f"center=({center[0]:.4f},{center[1]:.4f},{center[2]:.4f}), "
        f"size=({size[0]:.4f},{size[1]:.4f},{size[2]:.4f})"
    )
    return center, marker_path


def get_conveyor_max_box_top_z_155(stage, attached_box_path=None):
    """컨베이어 위 OriBoxA_*들의 최고 top_z를 읽는다. 실패하면 None."""
    if not bool(globals().get("CONVEYOR_SAFE_LIFT_ENABLED_155", False)):
        return None, "disabled"
    prefixes = tuple(str(p) for p in globals().get("CONVEYOR_SAFE_LIFT_INCLUDE_PREFIXES_155", ("/World/OriBoxA_",)))
    exclude_attached = bool(globals().get("CONVEYOR_SAFE_LIFT_EXCLUDE_ATTACHED_155", False))
    best_top = None
    best_path = None
    rows = []
    try:
        for prim in stage.Traverse():
            try:
                p = str(prim.GetPath())
            except Exception:
                continue
            if not any(p.startswith(pref) for pref in prefixes):
                continue
            # root만 본다. /World/OriBoxA_02/Small_Cardboard_box 같은 child는 제외.
            if p.count("/") != 2:
                continue
            if exclude_attached and attached_box_path and p == str(attached_box_path):
                continue
            bbox = get_world_bbox_info(stage, p)
            if bbox is None:
                continue
            top = float(bbox["max"][2])
            rows.append((p, top))
            if best_top is None or top > best_top:
                best_top = top
                best_path = p
    except Exception as e:
        return None, f"scan failed: {type(e).__name__}: {e}"
    if best_top is None:
        return None, "no OriBox top found"
    rows_txt = ", ".join([f"{pp.split('/')[-1]}:{zz:.3f}" for pp, zz in rows[:8]])
    return float(best_top), f"max_top={best_top:.4f} from {best_path}; boxes=[{rows_txt}]"

def lower_stack_support_cube_for_next_layer(stage, box_height, lower_count):
    """1층 2개 완료 후 BoxAprop를 박스 높이만큼 낮춘다."""
    if not STACK_LOWER_SUPPORT_AFTER_EACH_LAYER:
        return int(lower_count)
    lower_count = int(lower_count)
    if lower_count >= int(STACK_LOWER_MAX_LAYERS):
        return lower_count

    bbox = get_world_bbox_info(stage, STACK_SUPPORT_CUBE_PATH)
    if bbox is None:
        print(f"  [STACK_LOWER_SKIP] support cube not found: {STACK_SUPPORT_CUBE_PATH}")
        return lower_count

    box_height = max(float(box_height), 0.05)
    dz = box_height + float(STACK_LOWER_EXTRA_Z)
    cur_center = np.array(bbox["center"], dtype=float)
    new_center = cur_center.copy()
    new_center[2] -= dz

    ok = set_prim_world_translation(stage, STACK_SUPPORT_CUBE_PATH, new_center)
    for _ in range(10):
        simulation_app.update()

    new_bbox = get_world_bbox_info(stage, STACK_SUPPORT_CUBE_PATH)
    new_top = float(new_bbox["max"][2]) if new_bbox is not None else float("nan")
    print(
        f"  [STACK_PLATFORM_LOWER] layer_done={lower_count + 1}/{STACK_LOWER_MAX_LAYERS}, "
        f"dz={dz:.4f}, ok={ok}, "
        f"center_before=({cur_center[0]:.3f},{cur_center[1]:.3f},{cur_center[2]:.4f}), "
        f"center_after=({new_center[0]:.3f},{new_center[1]:.3f},{new_center[2]:.4f}), "
        f"new_top_z={new_top:.4f}"
    )
    return lower_count + 1


def get_robot_center_for_goal(stage):
    """로봇 중심/발판 중심 후보를 순서대로 찾아 반대편 좌표 계산에 사용한다."""
    for p in ROBOT_CENTER_CANDIDATE_PATHS:
        pos = get_world_translation(stage, p)
        if pos is not None and np.all(np.isfinite(pos)):
            return np.array(pos, dtype=float), p
    return None, None


def compute_mirror_goal_center(stage, box_center):
    """
    23_ 목표점 계산:
    - 박스가 컨베이어를 타고 멈춘 현재 box_center를 기준으로 한다.
    - 로봇 중심/발판 중심 후보를 읽어서 XY를 대칭 이동한다.
      goal_xy = 2*robot_xy - box_xy
    - 단, 완전 대칭점이 너무 멀면 로봇이 못 가고 release가 안 되므로 이동 거리를 제한한다.
    - Z는 실제 place 바닥 높이가 아니라 custom carry에서 별도 제어한다.
    """
    box_center = np.array(box_center, dtype=float)
    goal = box_center.copy()

    robot_center, center_path = get_robot_center_for_goal(stage)
    if robot_center is None:
        goal = box_center + np.array(GOAL_OFFSET_FROM_BOX_CENTER, dtype=float)
        goal[2] = box_center[2]
        print(f"  [WARN] robot center를 읽지 못해서 fallback goal offset 사용: goal={goal}")
        return goal

    raw_goal = box_center.copy()
    raw_goal[0] = 2.0 * robot_center[0] - box_center[0]
    raw_goal[1] = 2.0 * robot_center[1] - box_center[1]
    raw_goal[2] = box_center[2]

    delta_xy = raw_goal[:2] - box_center[:2]
    dist_xy = float(np.linalg.norm(delta_xy))
    max_dist = float(MIRROR_GOAL_MAX_XY_DISTANCE_FROM_PICK)
    goal[:] = raw_goal
    if max_dist > 0.0 and dist_xy > max_dist:
        scale = max_dist / max(dist_xy, 1e-9)
        goal[:2] = box_center[:2] + delta_xy * scale
        print(
            f"  [GOAL_CLAMP] mirror target distance {dist_xy:.3f}m > {max_dist:.3f}m. "
            f"목표를 로봇 도달 가능한 거리로 제한한다."
        )

    print(
        f"  [GOAL] mode={GOAL_MODE}, center_path={center_path}, "
        f"robot_center=({robot_center[0]:.3f},{robot_center[1]:.3f},{robot_center[2]:.3f}), "
        f"box_center=({box_center[0]:.3f},{box_center[1]:.3f},{box_center[2]:.3f}), "
        f"raw_goal=({raw_goal[0]:.3f},{raw_goal[1]:.3f},{raw_goal[2]:.3f}), "
        f"goal_center=({goal[0]:.3f},{goal[1]:.3f},{goal[2]:.3f})"
    )
    return goal

def set_task_pick_from_current_box(task, stage, bbox_info):
    """
    박스가 멈춘 순간의 bbox를 기준으로 pick 위치를 다시 확정한다.

    13_ 수정:
    - box 초기 원점/root가 바닥 기준이어도 bbox center/top_center를 다시 계산한다.
    - 로봇 controller에 넣는 picking_position은 bbox center가 아니라
      bbox top_center + BOX_PICK_TOP_CLEARANCE 를 사용한다.
    - 실제 scripted follow/release는 bbox center 기준으로 유지한다.
    """
    if bbox_info is None:
        return False

    center = np.array(bbox_info["center"], dtype=float)
    top_center = np.array(bbox_info["top_center"], dtype=float)
    height = float(bbox_info.get("height", top_center[2] - center[2]))

    root_pos = get_world_translation(stage, task.box_move_path)
    if root_pos is None:
        root_pos = center.copy()

    task._box_initial_root_pos = np.array(root_pos, dtype=float)
    task._box_initial_center = center.copy()
    task._box_initial_top_center = top_center.copy()
    task._box_height = height
    task._box_root_to_center_offset = task._box_initial_root_pos - task._box_initial_center
    try:
        root_ref = getattr(task, "active_box_root_path", None)
        if root_ref:
            root_ref_pos = get_world_translation(stage, root_ref)
            if root_ref_pos is not None:
                sep = float(np.linalg.norm(np.array(root_ref_pos, dtype=float) - np.array(task._box_initial_center, dtype=float)))
                print(
                    f"  [ROOT_CHILD_CHECK] root={root_ref}, move={task.box_move_path}, "
                    f"root_to_bbox_center={sep:.4f}m, move_path_is_child={task.box_move_path == task.box_path}"
                )
    except Exception:
        pass

    if USE_BOX_TOP_CENTER_AS_PICK_TARGET:
        task._pick_center = top_center.copy()
        task._pick_center[2] += float(BOX_PICK_TOP_CLEARANCE)
    else:
        task._pick_center = center.copy()

    # 16_ 핵심: M0609/발판 위치 변경 후 실제 VGC10 suction point가 박스보다
    # X+ / Y- 방향으로 빗나가므로 picking_position 자체를 보정한다.
    task._pick_center = np.array(task._pick_center, dtype=float) + np.array(PICK_TARGET_MANUAL_OFFSET, dtype=float)

    # 47_: goal은 기본 mirror 대신 BoxAprop 위 slot 좌표로 계산한다.
    # 박스는 흡착 전에는 절대 코드로 옮기지 않고, 감지 순간의 bbox를 기준으로만 목표를 계산한다.
    if MULTI_ORIBOX_STACKING_ENABLED:
        slot_index = int(getattr(task, "_stack_slot_index", 0))
        task._goal_center = compute_stack_goal_center(stage, bbox_info, slot_index)
    else:
        task._goal_center = compute_mirror_goal_center(stage, task._box_initial_center)
    return True


def move_box_center_to(stage, prim_path, desired_center, root_to_center_offset):
    """
    bbox center를 desired_center로 보내기 위해 root world translate를 계산해서 이동한다.
    root_to_center_offset = root_world_pos - bbox_center
    """
    desired_center = np.array(desired_center, dtype=float)
    desired_root = desired_center + np.array(root_to_center_offset, dtype=float)
    ok = set_prim_world_translation(stage, prim_path, desired_root)
    zero_subtree_velocity(stage, prim_path)
    return ok

def probe_move_path_affects_box_bbox(stage, move_path, box_path, delta_z=MOTION_PROBE_DELTA_Z):
    """
    move_path를 살짝 들어올렸을 때 box_path의 bbox가 실제로 따라오는지 확인한다.
    이 테스트가 실패하면 흡착 판정이 surface=ON이어도 박스가 안 들린다.
    """
    print("\n" + "=" * 60)
    print("[14.PROBE] 이동 prim이 실제 박스 bbox를 움직이는지 검사")
    print("=" * 60)
    print(f"  box_path  = {box_path}")
    print(f"  move_path = {move_path}")

    before_bbox = get_world_bbox_info(stage, box_path)
    before_move_pos = get_world_translation(stage, move_path)
    if before_bbox is None or before_move_pos is None:
        print("  [PROBE_FAIL] bbox 또는 move_path world translation을 읽지 못함")
        return False

    before_center = np.array(before_bbox["center"], dtype=float)
    target_move_pos = np.array(before_move_pos, dtype=float) + np.array([0.0, 0.0, float(delta_z)], dtype=float)
    ok_set = set_prim_world_translation(stage, move_path, target_move_pos)
    for _ in range(5):
        simulation_app.update()

    after_bbox = get_world_bbox_info(stage, box_path)
    after_center = np.array(after_bbox["center"], dtype=float) if after_bbox is not None else before_center.copy()
    moved = float(np.linalg.norm(after_center - before_center))
    moved_z = float(after_center[2] - before_center[2])

    # 원위치 복구
    set_prim_world_translation(stage, move_path, before_move_pos)
    for _ in range(5):
        simulation_app.update()

    success = bool(ok_set and abs(moved_z) > abs(delta_z) * 0.5)
    print(f"  [PROBE_RESULT] ok_set={ok_set}, bbox_moved={moved:.5f}m, bbox_moved_z={moved_z:.5f}m, expected_z≈{delta_z:.5f}m, success={success}")
    if not success:
        print("  [PROBE_HINT] 이 move_path로 움직여도 박스 bbox가 안 따라온다. BOX_MOVE_PRIM_PATH를 부모 prim으로 바꿔야 한다.")
    else:
        print("  [PROBE_OK] scripted suction에서 이 move_path를 움직이면 박스가 실제로 따라온다.")
    return success


def should_attach_oribox(event, suction_pos, bbox_info, box_stopped=True):
    """
    51_ 수정: 단일 suction point 대신 3x3 흡착점 그리드로 흡착 여부를 판단한다.

    조건:
    - 작업 영역에 들어온 OriBoxA_ 박스의 윗면 bbox 기준
    - 9개 점 중 SUCTION_GRID_ATTACH_MIN_POINTS개 이상이 윗면 안쪽에 있어야 함
    - 기본적으로 중심점 p11도 윗면 안쪽에 있어야 함
    """
    if bbox_info is None:
        return False, "grid_no_bbox"

    if WAIT_UNTIL_BOX_STOPPED_BEFORE_PICK and not box_stopped:
        return False, "box_not_stopped"

    stage = omni.usd.get_context().get_stage()

    if event in PICK_CLOSE_EVENTS:
        ok, summary, _info = evaluate_suction_grid_on_box_top(stage, bbox_info, event=event, verbose=False)
        return ok, summary

    if event in RETRY_CLOSE_EVENTS:
        ok, summary, _info = evaluate_suction_grid_on_box_top(stage, bbox_info, event=event, verbose=True)
        return ok, "retry_" + summary

    return False, "not_pick_event"


# ============================================================
# Task
# ============================================================
class M0609ConveyorBoxTask(BaseTask):

    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._task_achieved = False
        self._box_path = BOX_PRIM_PATH
        self._box_move_path = BOX_MOVE_PRIM_PATH
        self._stop_check_path = STOP_CHECK_PRIM_PATH
        self._active_box_root_name = None
        self._active_box_root_path = None  # 66_: 완료/skip 기준 root path 보관
        self._stack_slot_index = 0
        self._box_initial_root_pos = None
        self._box_initial_center = None
        self._box_initial_top_center = None
        self._box_height = None
        self._pick_center = None
        self._box_root_to_center_offset = None
        self._goal_center = None

    @property
    def box_path(self):
        return self._box_path

    @property
    def box_move_path(self):
        return self._box_move_path

    @property
    def active_box_root_path(self):
        return self._active_box_root_path

    def set_active_box(self, box_root_path, box_path=None, stack_slot_index=None):
        """현재 pick 대상으로 사용할 OriBoxA_*/OriBoxB_* root를 갱신한다.

        72_ 핵심:
        - USD에서 root에 Rigid Body를 붙이고 child는 Collider만 남긴 구조를 사용한다.
        - 완료/후보 제외, bbox/흡착/정지 판정, carry/follow 이동 모두 root 기준으로 통일한다.
        - box_path 인자는 과거 child 기반 코드 호환용으로만 받고 실제 기준에는 사용하지 않는다.
        """
        root_path = str(box_root_path)
        self._active_box_root_path = root_path
        self._box_move_path = root_path
        self._box_path = root_path
        self._stop_check_path = root_path
        self._active_box_root_name = root_path.rsplit("/", 1)[-1]
        if stack_slot_index is not None:
            self._stack_slot_index = int(stack_slot_index)
        self._box_initial_root_pos = None
        self._box_initial_center = None
        self._box_initial_top_center = None
        self._box_height = None
        self._pick_center = None
        self._box_root_to_center_offset = None
        self._goal_center = None

    @property
    def stop_check_path(self):
        return self._stop_check_path

    @property
    def goal_center(self):
        return self._goal_center

    @property
    def box_initial_center(self):
        return self._box_initial_center

    @property
    def box_initial_top_center(self):
        return self._box_initial_top_center

    @property
    def pick_center(self):
        return self._pick_center if self._pick_center is not None else self._box_initial_center

    @property
    def box_height(self):
        return self._box_height

    @property
    def box_root_to_center_offset(self):
        return self._box_root_to_center_offset

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._load_usd()
        self._remove_old_gripper()
        self._discover_links()
        self._attach_vgc10()
        self._attach_idle_vgc10_robots()
        self._setup_physics()
        self._register_robot(scene)
        self._setup_existing_box(scene)
        self._setup_vgc10_suction_anchor(scene)
        print("\n  [완료] Conveyor + OriBoxA 씬 구성 성공!\n")

    def _load_usd(self):
        print("\n" + "=" * 60)
        print("[1.LOAD] Conveyor USD 로드")
        print("=" * 60)

        usd_file = Path(USD_PATH)
        if not usd_file.exists():
            raise RuntimeError(f"Conveyor USD 파일이 없습니다: {USD_PATH}")

        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim.IsValid():
            world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()

        world_prim.GetReferences().AddReference(USD_PATH)

        # 147_: 사용자가 USD에서 /World 밖(root)에 둔 /Environment는
        # /World prim에 USD를 reference하면 defaultPrim(/World)만 들어오면서 빠질 수 있다.
        # 그래서 원본 USD의 /Environment prim을 현재 stage의 /Environment로 별도 reference한다.
        try:
            env_prim = stage.GetPrimAtPath("/Environment")
            if not env_prim or not env_prim.IsValid():
                env_prim = UsdGeom.Xform.Define(stage, "/Environment").GetPrim()
            env_prim.GetReferences().AddReference(USD_PATH, "/Environment")
            print("  [ENV_REF_147] root /Environment 별도 reference 추가: /Environment <= USD:/Environment")
        except Exception as _env_ref_e:
            print(f"  [ENV_REF_147][WARN] /Environment 별도 reference 실패: {_env_ref_e}")

        for _ in range(40):
            simulation_app.update()

        # 147_: reference composition 후 root /Environment 및 하위 prim을 active/visible로 강제 복구한다.
        force_show_environment_prims_146(stage)

        # 29_: USD 안에 저장된 goal_marker가 있어도 실행 시 보이지 않게 정리한다.
        if not SHOW_GOAL_MARKER:
            remove_goal_marker(stage)

        # 106_: rsd455 prim은 유지하고, robot 하위 legacy camera_graph만 필요 시 비활성화한다.
        disable_legacy_camera_graphs(stage)

        # 109_: 사진으로 보낸 Cube(/Cube 또는 /World/Cube)가 실행 후에도 화면에 보이게 유지한다.
        force_show_cube_prims_109(stage)

        # 146_: 새 USD에서 Prim Path가 Environment인 환경 오브젝트들이 사라지지 않게 강제 표시한다.
        force_show_environment_prims_146(stage)

        # 77_: pick zone visual은 생성하지 않는다. 감지 계산은 PICK_ZONE_CENTER/SIZE 숫자만 사용한다.
        create_pick_zone_visual(stage)

        global ROBOT_PRIM_PATH, OLD_GRIPPER_PRIM_PATH, VGC10_FOLLOW_TARGET_PATH

        resolved_robot_path = _resolve_robot_prim_from_root(stage, ACTIVE_ROBOT_ROOT_PATH)
        if resolved_robot_path is not None:
            ROBOT_PRIM_PATH = resolved_robot_path
            OLD_GRIPPER_PRIM_PATH = ROBOT_PRIM_PATH + "/onrobot_rg2ft"
            resolved_tool_target = _find_tool_target_for_robot_prim(stage, ROBOT_PRIM_PATH)
            if resolved_tool_target is not None:
                VGC10_FOLLOW_TARGET_PATH = resolved_tool_target

        robot_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        if not robot_prim.IsValid():
            candidates = []
            try:
                world = stage.GetPrimAtPath("/World")
                if world.IsValid():
                    for child in world.GetChildren():
                        if child.GetName().startswith(IDLE_M0609_ROOT_PREFIX) or "m0609" in child.GetName().lower():
                            candidates.append(str(child.GetPath()))
            except Exception:
                pass
            raise RuntimeError(
                f"Conveyor USD는 열었지만 active 로봇 prim을 찾지 못했습니다.\n"
                f"  ACTIVE_ROBOT_ROOT_PATH={ACTIVE_ROBOT_ROOT_PATH}\n"
                f"  ROBOT_PRIM_PATH={ROBOT_PRIM_PATH}\n"
                f"  USD_PATH={USD_PATH}\n"
                f"  m0609 candidates={candidates}\n"
                f"Stage에서 실제 로봇 경로를 확인해서 ACTIVE_ROBOT_ROOT_PATH를 수정하세요."
            )

        print(f"  [OK] {USD_PATH}")
        print(f"  [OK] active robot root = {ACTIVE_ROBOT_ROOT_PATH}")
        print(f"  [OK] active robot prim = {ROBOT_PRIM_PATH}")
        print(f"  [OK] active VGC10 target = {VGC10_FOLLOW_TARGET_PATH}")

    def _remove_old_gripper(self):
        print("\n" + "=" * 60)
        print("[1-1.REMOVE] 기존 RG2 집게 삭제")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()
        remove_old_gripper(stage)
        for _ in range(5):
            simulation_app.update()

    def _discover_links(self):
        print("\n" + "=" * 60)
        print("[2.DISCOVER] 링크 경로 탐색")
        print("=" * 60)
        self._ee_path = find_prim_path_by_name(ROBOT_PRIM_PATH, EE_LINK_NAME)
        if self._ee_path is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found under {ROBOT_PRIM_PATH}")
        print(f"  EE ({EE_LINK_NAME}) = {self._ee_path}")

    def _attach_vgc10(self):
        print("\n" + "=" * 60)
        print("[2-1.ATTACH] VGC10 Vacuum Gripper 부착 - active m0609_A")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()
        attach_vgc10_to_link6(stage)
        for _ in range(5):
            simulation_app.update()

    def _attach_idle_vgc10_robots(self):
        print("\n" + "=" * 60)
        print("[2-2.IDLE] 다른 m0609_ 로봇 VGC10 visual 부착 / 제어 없음")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()
        self._idle_vgc10_attached = attach_vgc10_to_idle_m0609_robots(stage)
        for _ in range(5):
            simulation_app.update()

    def _setup_physics(self):
        print("\n" + "=" * 60)
        print("[3.PHYSICS] 물리 설정")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()

        drive_count = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH)):
            for dt in ["angular", "linear"]:
                drive = UsdPhysics.DriveAPI.Get(prim, dt)
                if drive:
                    drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                    drive_count += 1
        print(f"  [OK] drive updated: {drive_count}")

    def _register_robot(self, scene):
        print("\n" + "=" * 60)
        print("[4.REGISTER] 로봇 등록")
        print("=" * 60)
        self._robot = scene.add(
            SingleManipulator(
                prim_path=ROBOT_PRIM_PATH,
                name="m0609_robot",
                end_effector_prim_path=self._ee_path,
                gripper=None,
            )
        )
        print(f"  [OK] SingleManipulator without RG2 gripper: {ROBOT_PRIM_PATH}")

    def _setup_existing_box(self, scene):
        print("\n" + "=" * 60)
        print("[5.SCENE] OriBoxA_* 다중 박스 대상 등록 / 첫 후보 확인")
        print("=" * 60)

        stage = omni.usd.get_context().get_stage()

        # 63_: /World 하위의 OriBoxA_01, OriBoxA_02, ... 후보를 우선 사용한다.
        candidates = discover_oriboxa_stack_candidates(stage, completed_roots=set()) if MULTI_ORIBOX_STACKING_ENABLED else []
        if candidates:
            first = candidates[0]
            self.set_active_box(first["root_path"], first["box_path"], stack_slot_index=0)
            print(f"  [OK] 초기 후보 box root = {first['root_path']}")
            print(f"  [OK] 초기 후보 box path = {first['box_path']}")
        else:
            # 72_: 호환 fallback도 root 기준 단일 OriBox를 사용한다.
            self._box_path = ensure_oribox_a_exists(stage)
            if not stage.GetPrimAtPath(self._box_path).IsValid():
                raise RuntimeError(
                    f"직접 배치된 OriBox root를 찾지 못했습니다. GUI에서 /World/OriBoxA_01 root가 있는지 확인하세요.\n"
                    f"  required root={BOX_ROOT_PATH}"
                )
            self.set_active_box(BOX_ROOT_PATH, self._box_path, stack_slot_index=0)
            print("  [WARN] /World/OriBoxA_/OriBoxB 후보가 없어 단일 BOX_ROOT_PATH fallback 사용")

        # 65_: robotAprop_01 / OriBoxA_* collider 상태를 실행 중 Stage에서 명시적으로 보정한다.
        # 주의: 이 보정은 물리 충돌용 collider를 켜는 것이고, top-lock으로 직접 Xform 이동 중인 박스가
        # collider에 의해 자동으로 멈추는 것은 아니다. 그래서 아래 make_custom_carry_targets()에서
        # robotAprop_01 위를 충분히 높은 Z로 지나가도록 추가 보정도 한다.
        repair_robotaprop_and_oribox_colliders(stage, completed_roots=set(), verbose=True)
        # 68_: 실행 시작 시점에 USD에서 맞춘 root/child 중심이 벌어져 있으면 다시 맞춘다.
        sync_oribox_child_centers_with_roots(stage, verbose=True)

        # 77_: USD 직접 편집 환경만 사용한다. 예전 코드처럼 default ground plane을 새로 만들지 않는다.
        if bool(globals().get("ADD_DEFAULT_GROUND_PLANE", False)):
            try:
                scene.add_default_ground_plane()
                print("  [OK] default ground plane")
            except Exception as exc:
                print(f"  [WARN] ground plane 추가 실패 또는 이미 존재: {exc}")
        else:
            print("  [CLEAN] default ground plane 생성 안 함: USD 환경 그대로 사용")

        bbox = get_world_bbox_info(stage, self._box_path)
        if bbox is None:
            raise RuntimeError(f"박스 bbox를 계산하지 못했습니다: {self._box_path}")

        root_pos = get_world_translation(stage, self._box_move_path)
        if root_pos is None:
            root_pos = bbox["center"].copy()

        self._box_initial_root_pos = np.array(root_pos, dtype=float)
        self._box_initial_center = np.array(bbox["center"], dtype=float)
        self._box_initial_top_center = np.array(bbox["top_center"], dtype=float)
        self._box_height = float(bbox.get("height", bbox["size"][2]))
        self._box_root_to_center_offset = self._box_initial_root_pos - self._box_initial_center

        if USE_BOX_TOP_CENTER_AS_PICK_TARGET:
            self._pick_center = self._box_initial_top_center.copy()
            self._pick_center[2] += float(BOX_PICK_TOP_CLEARANCE)
        else:
            self._pick_center = self._box_initial_center.copy()
        self._pick_center = np.array(self._pick_center, dtype=float) + np.array(PICK_TARGET_MANUAL_OFFSET, dtype=float)

        if MULTI_ORIBOX_STACKING_ENABLED:
            self._goal_center = compute_stack_goal_center(stage, bbox, int(self._stack_slot_index))
        else:
            self._goal_center = compute_mirror_goal_center(stage, self._box_initial_center)

        if not stage.GetPrimAtPath(self._stop_check_path).IsValid():
            raise RuntimeError(f"정지/영역 판정 대상 prim을 찾지 못했습니다: {self._stop_check_path}")

        print(f"  [OK] box path        = {self._box_path}")
        print(f"  [OK] box move path   = {self._box_move_path}")
        print(f"  [OK] stop check path = {self._stop_check_path}")
        print(f"  [OK] stack slot      = {self._stack_slot_index + 1}/{STACK_SLOT_COUNT}")
        print(f"  [OK] root world pos  = {self._box_initial_root_pos}")
        print(f"  [OK] bbox center     = {self._box_initial_center}")
        print(f"  [OK] bbox size       = {bbox['size']}")
        print(f"  [OK] bbox top center = {bbox['top_center']}")
        print(f"  [OK] bbox height     = {self._box_height:.5f} m")
        print(f"  [OK] pick target     = {self._pick_center}  # top_center + {BOX_PICK_TOP_CLEARANCE:.3f}m + manual_offset {PICK_TARGET_MANUAL_OFFSET}")
        print(f"  [OK] goal center     = {self._goal_center}")

        remove_goal_marker(stage)

        # 처음에는 일반 물리 상태로 둔다.
        set_prim_kinematic(stage, self._box_path, False)
        zero_prim_velocity(stage, self._box_path)
        if DEBUG_MOVE_PARENT_AND_FOLLOW:
            probe_move_path_affects_box_bbox(stage, self._box_move_path, self._box_path)

    def _setup_vgc10_suction_anchor(self, scene):
        print("\n" + "=" * 60)
        print("[5-1.VGC10 SUCTION] VGC10 기준 흡착점 활성화")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()

        clear_prim_if_exists(stage, SUCTION_BODY_PATH)
        clear_prim_if_exists(stage, SUCTION_MARKER_PATH)
        clear_prim_if_exists(stage, VGC10_SUCTION_DEBUG_MARKER_PATH)

        if not stage.GetPrimAtPath(VGC10_SUCTION_POINT_PATH).IsValid():
            suction_point_xform = UsdGeom.Xform.Define(stage, VGC10_SUCTION_POINT_PATH)
            suction_point_prim = suction_point_xform.GetPrim()
            suction_mat = Gf.Matrix4d(1.0)
            suction_mat.SetTranslateOnly(Gf.Vec3d(
                float(VGC10_SUCTION_LOCAL_OFFSET[0]),
                float(VGC10_SUCTION_LOCAL_OFFSET[1]),
                float(VGC10_SUCTION_LOCAL_OFFSET[2]),
            ))
            set_prim_local_matrix(suction_point_prim, suction_mat)

        print("  [OK] scripted_suction_body 제거")
        print("  [OK] scripted_suction_direction_marker 제거")
        print(f"  [OK] VGC10 suction point 활성화: {VGC10_SUCTION_POINT_PATH}")
        print(f"       offset = {VGC10_SUCTION_LOCAL_OFFSET}")
        print("       화면에 흡착점이 필요하면 DEBUG_SHOW_SUCTION_POINT=True 로 바꿔라.")

    def get_observations(self):
        stage = omni.usd.get_context().get_stage()
        bbox = get_world_bbox_info(stage, self._box_path)
        if bbox is None:
            box_center = self._box_initial_center.copy()
            box_top_center = self._box_initial_center.copy()
        else:
            box_center = bbox["center"]
            box_top_center = bbox["top_center"]

        return {
            self._robot.name: {
                "joint_positions": self._robot.get_joint_positions(),
            },
            "oribox": {
                "position": box_center,
                "top_center": box_top_center,
                "goal_position": self._goal_center,
            },
        }

    def pre_step(self, control_index, simulation_time):
        if self._goal_center is None:
            return
        stage = omni.usd.get_context().get_stage()
        bbox = get_world_bbox_info(stage, self._box_path)
        if bbox is None:
            return
        if not self._task_achieved and np.linalg.norm(self._goal_center - bbox["center"]) < 0.08:
            self._task_achieved = True

    def post_reset(self):
        stage = omni.usd.get_context().get_stage()
        # 19_: 박스 위치는 코드로 되돌리지 않는다. 컨베이어/물리 흐름에 맡긴다.
        if RESET_BOX_TO_USD_START_ON_PLAY and self._box_initial_root_pos is not None:
            set_prim_world_translation(stage, self._box_move_path, self._box_initial_root_pos)
        set_prim_kinematic(stage, self._box_path, False)
        zero_prim_velocity(stage, self._box_path)
        self._task_achieved = False




def _vec3(arr):
    arr = np.array(arr, dtype=float)
    return np.array([float(arr[0]), float(arr[1]), float(arr[2])], dtype=float)


def _quat_np(q):
    try:
        return np.array(q, dtype=float)
    except Exception:
        return q


def get_current_ee_pose(robot):
    ee_pos, ee_quat = robot.end_effector.get_world_pose()
    return _vec3(ee_pos), ee_quat


def make_ee_target_for_desired_suction(robot, current_suction_pos, desired_suction_pos, fixed_orientation=None):
    """
    RMPFlow는 link_6 EE를 움직이고, 실제 흡착점은 VGC10 suction point다.
    현재 EE와 suction point 사이의 world offset을 유지한 채 desired_suction_pos가 되도록 EE target을 계산한다.

    fixed_orientation=None이면 orientation constraint를 걸지 않는다.
    55_에서는 pre-align 중 하늘 보는 자세가 고정되는 문제를 피하기 위해 None을 사용한다.
    """
    ee_pos, ee_quat = get_current_ee_pose(robot)
    current_suction_pos = _vec3(current_suction_pos)
    desired_suction_pos = _vec3(desired_suction_pos)
    ee_to_suction = current_suction_pos - ee_pos
    target_ee_pos = desired_suction_pos - ee_to_suction
    target_ee_quat = fixed_orientation
    return target_ee_pos, target_ee_quat


def apply_cartesian_suction_target(cart_controller, robot, current_suction_pos, desired_suction_pos, fixed_orientation=None):
    target_ee_pos, target_ee_quat = make_ee_target_for_desired_suction(
        robot,
        current_suction_pos=current_suction_pos,
        desired_suction_pos=desired_suction_pos,
        fixed_orientation=fixed_orientation,
    )
    kwargs = {"target_end_effector_position": target_ee_pos}
    if target_ee_quat is not None:
        kwargs["target_end_effector_orientation"] = target_ee_quat
    action = cart_controller.forward(**kwargs)
    robot.apply_action(action)
    return target_ee_pos


def _norm2(v, fallback=None):
    v = np.array(v, dtype=float)[:2]
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        if fallback is None:
            return np.array([1.0, 0.0], dtype=float)
        f = np.array(fallback, dtype=float)[:2]
        fn = float(np.linalg.norm(f))
        return f / max(fn, 1e-9)
    return v / n


def _push_xy_outside_robot_guard(stage, xy, center_xy=None):
    """목표 XY가 발판/로봇 중심 금지 반경 안이면 바깥으로 밀어낸다."""
    xy = np.array(xy, dtype=float)[:2]
    if not CARRY_AVOID_BASE_AND_ARM:
        return xy
    if center_xy is None:
        robot_center, _ = get_robot_center_for_goal(stage)
        if robot_center is None:
            return xy
        center_xy = np.array(robot_center, dtype=float)[:2]
    else:
        center_xy = np.array(center_xy, dtype=float)[:2]

    guard = float(CARRY_FORBIDDEN_RADIUS_XY + CARRY_ROUTE_MARGIN_XY)
    v = xy - center_xy
    d = float(np.linalg.norm(v))
    if d < guard:
        v = _norm2(v, fallback=np.array([0.0, -1.0]))
        xy = center_xy + v * guard
        print(f"  [AVOID_GOAL] place xy was inside robot/base guard. pushed outside to ({xy[0]:.3f},{xy[1]:.3f}), guard={guard:.3f}")
    return xy


def _make_avoidance_suction_waypoints(stage, attach_suction_pos, goal_suction_pos):
    """
    시작 suction XY에서 목표 suction XY까지 직선으로 가지 않고,
    로봇/발판 중심 주변 금지 원을 옆으로 돌아가는 waypoint를 만든다.
    """
    start = np.array(attach_suction_pos, dtype=float)
    goal = np.array(goal_suction_pos, dtype=float)
    safe_z = float(max(start[2], goal[2], CUSTOM_CARRY_SAFE_SUCTION_Z))
    start[2] = safe_z
    goal[2] = safe_z

    robot_center, center_path = get_robot_center_for_goal(stage)
    if (not CARRY_AVOID_BASE_AND_ARM) or robot_center is None:
        return [
            {"name": "lift", "kind": "lift", "target": start.copy()},
            {"name": "move", "kind": "move", "target": goal.copy()},
        ]

    center_xy = np.array(robot_center, dtype=float)[:2]
    start_xy = start[:2]
    goal_xy = goal[:2]

    main_dir = _norm2(goal_xy - start_xy, fallback=start_xy - center_xy)
    tangent = np.array([-main_dir[1], main_dir[0]], dtype=float) * float(CARRY_ROUTE_SIDE_SIGN)
    guard = float(CARRY_FORBIDDEN_RADIUS_XY + CARRY_ROUTE_MARGIN_XY)

    # start/goal을 main_dir 축으로 투영하고, tangent 방향으로 guard만큼 빼서 중심을 돌아간다.
    start_d = float(np.dot(start_xy - center_xy, main_dir))
    goal_d = float(np.dot(goal_xy - center_xy, main_dir))

    via1_xy = center_xy + main_dir * start_d + tangent * guard
    via2_xy = center_xy + main_dir * goal_d + tangent * guard

    # waypoint가 너무 가까우면 생략해도 되지만, 로그/디버깅을 위해 그대로 둔다.
    via1 = np.array([via1_xy[0], via1_xy[1], safe_z], dtype=float)
    via2 = np.array([via2_xy[0], via2_xy[1], safe_z], dtype=float)

    print(
        f"  [AVOID_ROUTE] center_path={center_path}, center=({center_xy[0]:.3f},{center_xy[1]:.3f}), "
        f"guard={guard:.3f}, side={CARRY_ROUTE_SIDE_SIGN}, "
        f"start=({start_xy[0]:.3f},{start_xy[1]:.3f}), "
        f"via1=({via1[0]:.3f},{via1[1]:.3f}), via2=({via2[0]:.3f},{via2[1]:.3f}), "
        f"goal=({goal_xy[0]:.3f},{goal_xy[1]:.3f}), z={safe_z:.3f}"
    )

    return [
        {"name": "lift", "kind": "lift", "target": start.copy()},
        {"name": "avoid_start", "kind": "move", "target": via1},
        {"name": "avoid_goal", "kind": "move", "target": via2},
        {"name": "approach_goal", "kind": "move", "target": goal.copy()},
    ]



def _fit_joint_delta(joints, delta):
    joints = np.array(joints, dtype=float)
    delta = np.array(delta, dtype=float)
    if len(delta) < len(joints):
        delta = np.pad(delta, (0, len(joints) - len(delta)), mode="constant")
    elif len(delta) > len(joints):
        delta = delta[:len(joints)]
    return delta


def _clamp_first_joint_delta(delta):
    delta = float(delta)
    return float(np.clip(delta, -float(JOINT_SWING_CLAMP_RAD), float(JOINT_SWING_CLAMP_RAD)))


def _estimate_link2_orient_z_deg_from_j2(j2_rad, reference_j2_rad):
    """
    Stage GUI에서 보이는 초기 link_2 Orient Z=-90도를 기준으로 joint_2 변화량을 더해 추정한다.
    이 값은 USD Transform을 직접 쓰는 값이 아니라, joint_2 목표 제한을 위한 진단/가드 값이다.
    """
    sign = float(globals().get("LINK2_ORIENT_Z_SIGN", 1.0))
    initial_deg = float(globals().get("LINK2_ORIENT_Z_INITIAL_DEG", -90.0))
    return float(initial_deg + sign * np.degrees(float(j2_rad) - float(reference_j2_rad)))


def _link2_guard_j2_boundary_rad(reference_j2_rad):
    sign = float(globals().get("LINK2_ORIENT_Z_SIGN", 1.0))
    initial_deg = float(globals().get("LINK2_ORIENT_Z_INITIAL_DEG", -90.0))
    min_deg = float(globals().get("LINK2_ORIENT_Z_MIN_DEG", -90.0))
    if abs(sign) < 1.0e-9:
        return float(reference_j2_rad)
    return float(reference_j2_rad + np.radians((min_deg - initial_deg) / sign))


def _apply_link2_orient_z_guard_to_joints(joints, reference_j2_rad, label="target"):
    arr = np.array(joints, dtype=float).copy()
    if (not bool(globals().get("LINK2_ORIENT_Z_GUARD_ENABLED", False))) or len(arr) < 2:
        return arr
    sign = float(globals().get("LINK2_ORIENT_Z_SIGN", 1.0))
    min_deg = float(globals().get("LINK2_ORIENT_Z_MIN_DEG", -90.0))
    before_j2 = float(arr[1])
    before_est = _estimate_link2_orient_z_deg_from_j2(before_j2, reference_j2_rad)
    boundary_j2 = _link2_guard_j2_boundary_rad(reference_j2_rad)

    clamped = False
    if sign >= 0.0:
        if arr[1] < boundary_j2:
            arr[1] = boundary_j2
            clamped = True
    else:
        if arr[1] > boundary_j2:
            arr[1] = boundary_j2
            clamped = True

    after_est = _estimate_link2_orient_z_deg_from_j2(float(arr[1]), reference_j2_rad)
    if bool(globals().get("LINK2_ORIENT_Z_GUARD_LOG", True)):
        print(
            f"[LINK2_ORIENT_GUARD_108][{label}] "
            f"ref_j2={reference_j2_rad:+.6f} rad, boundary_j2={boundary_j2:+.6f} rad, "
            f"j2_before={before_j2:+.6f} rad -> j2_after={float(arr[1]):+.6f} rad, "
            f"link2_z_est_before={before_est:+.2f} deg, link2_z_est_after={after_est:+.2f} deg, "
            f"min={min_deg:+.2f} deg, clamped={clamped}"
        )
    return arr


def make_joint_swing_carry_targets(stage, robot, task, attach_center, attach_suction_pos, attached_center_offset):
    """
    78_ 관절 기반: 수직 lift 후 팔레타이징 이동은 joint_1(link_1 z축) 회전만 사용. 단, lift 자세를 더 펴서 회전 원호 반지름을 넓힌다.
    - link_2 transform을 직접 바꾸지 않고 joint_2/joint_3로 들어올린다.
    - 팔레트 slot 방향 이동은 joint_1 회전으로만 처리한다.
    - 박스는 매 step suction point + attached_center_offset만 따라온다.
    """
    start_joints = np.array(robot.get_joint_positions(), dtype=float)
    dof = len(start_joints)

    lift_delta = np.zeros(dof, dtype=float)
    if dof >= 2:
        lift_delta[1] = float(JOINT_LIFT_SIGN) * float(JOINT_LIFT_J2_DELTA_RAD)
    if dof >= 3 and JOINT_USE_J3_FOR_LIFT:
        lift_delta[2] = float(JOINT_LIFT_SIGN) * float(JOINT_LIFT_J3_DELTA_RAD)

    lift_joints = start_joints + lift_delta

    # 108_: link_2 Orient Z guard. 초기 link_2 Orient Z=-90도 기준을 joint_2 제한으로 변환한다.
    link2_j2_ref_rad = float(start_joints[1]) if dof >= 2 else 0.0
    lift_joints = _apply_link2_orient_z_guard_to_joints(lift_joints, link2_j2_ref_rad, label="lift_joints")

    swing_delta = np.zeros(dof, dtype=float)
    if dof >= 1:
        swing_delta[0] = _clamp_first_joint_delta(float(JOINT_SWING_SIGN) * float(JOINT_SWING_DELTA_RAD))
    swing_joints = lift_joints + swing_delta
    swing_joints = _apply_link2_orient_z_guard_to_joints(swing_joints, link2_j2_ref_rad, label="swing_joints_initial")

    lower_joints = swing_joints - lift_delta * float(JOINT_LOWER_RETURN_RATIO)
    lower_joints = _apply_link2_orient_z_guard_to_joints(lower_joints, link2_j2_ref_rad, label="lower_joints_initial")

    robot_center, center_path = get_robot_center_for_goal(stage)
    attach_center = _vec3(attach_center)
    attach_suction_pos = _vec3(attach_suction_pos)

    # 52_: 반 바퀴 고정 회전이 아니라, 현재 박스 위치 -> BoxAprop slot 목표 위치로 가는 joint_1 회전량을 계산한다.
    # 이렇게 해야 BoxAprop scale/위치가 USD에서 바뀌어도 1번 로봇이 slot 방향으로 회전한다.
    goal_center = np.array(getattr(task, "goal_center", attach_center), dtype=float)
    if robot_center is not None:
        rc = np.array(robot_center[:2], dtype=float)
        start_vec = attach_center[:2] - rc
        goal_vec = goal_center[:2] - rc
        start_angle = float(np.arctan2(start_vec[1], start_vec[0]))
        goal_angle = float(np.arctan2(goal_vec[1], goal_vec[0]))
        delta_angle = goal_angle - start_angle
        # [-pi, pi] 정규화
        delta_angle = float((delta_angle + np.pi) % (2.0 * np.pi) - np.pi)
        computed_swing_delta = _clamp_first_joint_delta(delta_angle)
        if dof >= 1:
            swing_delta[0] = computed_swing_delta
            swing_joints = lift_joints + swing_delta
            swing_joints = _apply_link2_orient_z_guard_to_joints(swing_joints, link2_j2_ref_rad, label="swing_joints_goal")
            lower_joints = swing_joints - lift_delta * float(JOINT_LOWER_RETURN_RATIO)
            lower_joints = _apply_link2_orient_z_guard_to_joints(lower_joints, link2_j2_ref_rad, label="lower_joints_goal")
        mirror_xy = goal_center[:2]
    else:
        mirror_xy = attach_center[:2] + np.array(GOAL_OFFSET_FROM_BOX_CENTER[:2], dtype=float)
        center_path = "fallback_offset"
        computed_swing_delta = float(swing_delta[0]) if dof >= 1 else 0.0

    print(
        f"  [JOINT_SWING_GOAL] center_path={center_path}, "
        f"attach_xy=({attach_center[0]:.3f},{attach_center[1]:.3f}), "
        f"goal_xy=({goal_center[0]:.3f},{goal_center[1]:.3f}), "
        f"j1_delta={computed_swing_delta:.3f} rad, wide_radius_j2={JOINT_LIFT_J2_DELTA_RAD:.3f}, wide_radius_j3={JOINT_LIFT_J3_DELTA_RAD:.3f}"
    )
    if dof >= 2:
        print(
            f"  [LINK2_ORIENT_CALC_108] initial_link2_z={LINK2_ORIENT_Z_INITIAL_DEG:+.1f}deg, "
            f"target_link2_z={LINK2_ORIENT_Z_TARGET_DEG:+.1f}deg, min_link2_z={LINK2_ORIENT_Z_MIN_DEG:+.1f}deg, ref_j2={link2_j2_ref_rad:+.6f}rad, "
            f"start_est={_estimate_link2_orient_z_deg_from_j2(start_joints[1], link2_j2_ref_rad):+.2f}deg, "
            f"lift_est={_estimate_link2_orient_z_deg_from_j2(lift_joints[1], link2_j2_ref_rad):+.2f}deg, "
            f"swing_est={_estimate_link2_orient_z_deg_from_j2(swing_joints[1], link2_j2_ref_rad):+.2f}deg"
        )

    # 103_: 진단 모드에서는 lift를 하지 않는다.
    # FixedJoint 생성만으로 박스가 튀는지 확인해야 하므로, 로봇 관절 목표는 현재 자세 그대로 유지한다.
    if bool(globals().get("PHYSICS_FIXED_JOINT_DIAGNOSTIC_NO_LIFT", False)):
        diag_steps = int(globals().get("PHYSICS_DIAGNOSTIC_HOLD_STEPS", 120))
        phase_sequence = [
            {"name": "joint_diagnostic_hold", "kind": "joint", "target_joints": start_joints, "steps": diag_steps},
        ]
        print(f"  [PHYSICS_JOINT_DIAG_106] phases=['joint_diagnostic_hold']; should not run in 106 because no_lift=False. hold={diag_steps} steps.")
    # 80_: 물리 FixedJoint 테스트에서는 팔레트 방향 swing을 하지 않는다.
    # 박스를 코드로 따라오게 하는지 여부가 아니라, 실제 joint 연결로 lift되는지만 먼저 확인한다.
    elif bool(globals().get("PHYSICS_FIXED_JOINT_LIFT_ONLY_TEST", False)):
        stabilize_steps = int(globals().get("PHYSICS_FIXED_JOINT_STABILIZE_STEPS", 70))
        phase_sequence = [
            {"name": "joint_attach_stabilize", "kind": "joint", "target_joints": start_joints, "steps": stabilize_steps},
            {"name": "joint_lift", "kind": "joint", "target_joints": lift_joints, "steps": int(JOINT_LIFT_STEPS)},
            {"name": "joint_hold", "kind": "joint", "target_joints": lift_joints, "steps": int(VERTICAL_LIFT_HOLD_STEPS)},
        ]
        print(f"  [PHYSICS_LIFT_ONLY_TEST_107] phases=['joint_attach_stabilize','joint_lift','joint_hold']; normally disabled in 107.")
    else:
        phase_sequence = [
            {"name": "joint_attach_stabilize", "kind": "joint", "target_joints": start_joints, "steps": int(PHYSICS_FIXED_JOINT_STABILIZE_STEPS)},
            {"name": "joint_lift", "kind": "joint", "target_joints": lift_joints, "steps": int(JOINT_LIFT_STEPS)},
            {"name": "joint_swing", "kind": "joint", "target_joints": swing_joints, "steps": int(JOINT_SWING_STEPS)},
            # 75_: 여기서 joint_lower/joint_settle을 제거한다.
            # 기존에는 swing 후 j2/j3를 다시 움직이면서 suction/box XY가 크게 튀었다.
            # 팔레타이징 이동은 link_1 z축 회전까지만 하고, 그 위치에서 그대로 release한다. snap/순간이동은 하지 않는다.
        ]

    return {
        "mode": "JOINT_SWING",
        "center_path": center_path,
        "start_joints": start_joints,
        "lift_delta": lift_delta,
        "swing_delta": swing_delta,
        "lift_joints": lift_joints,
        "swing_joints": swing_joints,
        "lower_joints": lower_joints,
        "mirror_xy_estimate": mirror_xy,
        "goal_center": goal_center,
        "computed_j1_delta": computed_swing_delta,
        "attach_center": attach_center,
        "attach_suction": attach_suction_pos,
        "attached_center_offset": _vec3(attached_center_offset),
        "phase_sequence": phase_sequence,
    }


def make_custom_carry_targets(stage, task, attach_center, attach_suction_pos, attached_center_offset):
    """
    24_ 순간이동/관통 방지용 target 생성.
    - 박스 target을 직접 set하지 않고 suction point target만 만든다.
    - 박스는 매 프레임 실제 suction point + attach offset만 따라간다.
    - 로봇/발판 큐브 주변은 waypoint로 돌아간다.
    """
    attach_center = _vec3(attach_center)
    attach_suction_pos = _vec3(attach_suction_pos)
    attached_center_offset = _vec3(attached_center_offset)

    # 60_: 먼저 실제 윗면 흡착 후 Z로 들어올리고, BoxAprop 위의 현재 slot 좌표로 이동한 뒤 살포시 내려놓는다.
    # release 직전까지 박스는 실제 suction 위치 + attach offset을 따라간다.
    if bool(VERTICAL_LIFT_ONLY_TEST):
        lift_suction = attach_suction_pos.copy()
        lift_suction[2] = float(attach_suction_pos[2]) + float(VERTICAL_LIFT_DELTA_Z)

        # 65_: robotAprop_01 같은 고정 장애물을 실제 물리로 밀고 지나가는 방식이 아니므로,
        # 박스 바닥이 장애물 윗면보다 충분히 높게 지나가도록 수직 리프트 높이를 자동 상향한다.
        clear_z, clear_reason = get_robotaprop_clearance_suction_z(stage, float(task.box_height), attached_center_offset)
        if clear_z is not None and float(lift_suction[2]) < float(clear_z):
            old_z = float(lift_suction[2])
            lift_suction[2] = float(clear_z)
            print(
                f"  [ROBOTAPROP_CLEARANCE] lift_suction_z raised {old_z:.4f} -> {lift_suction[2]:.4f}; {clear_reason}"
            )
        elif clear_z is not None:
            print(
                f"  [ROBOTAPROP_CLEARANCE] current lift_suction_z={float(lift_suction[2]):.4f} ok, required={float(clear_z):.4f}; {clear_reason}"
            )
        else:
            print(f"  [ROBOTAPROP_CLEARANCE] no z raise: {clear_reason}")

        # 155_: 컨베이어 위 다른 박스보다 충분히 높게 들어 올린 뒤에만 큰 회전/link_1 이동을 시작한다.
        conv_top_z_155, conv_reason_155 = get_conveyor_max_box_top_z_155(stage, attached_box_path=getattr(task, "box_path", None))
        if conv_top_z_155 is not None:
            # suction 목표 높이 기준: 다른 박스 최고 윗면 + 안전 여유.
            required_suction_z_155 = float(conv_top_z_155) + float(CONVEYOR_SAFE_LIFT_CLEARANCE_Z_155)
            if float(lift_suction[2]) < required_suction_z_155:
                old_z_155 = float(lift_suction[2])
                lift_suction[2] = required_suction_z_155
                print(
                    f"  [CONVEYOR_SAFE_LIFT_155] lift_suction_z raised {old_z_155:.4f} -> {lift_suction[2]:.4f}; {conv_reason_155}; clearance={CONVEYOR_SAFE_LIFT_CLEARANCE_Z_155:.3f}"
                )
            else:
                print(
                    f"  [CONVEYOR_SAFE_LIFT_155] current lift_suction_z={float(lift_suction[2]):.4f} ok; required={required_suction_z_155:.4f}; {conv_reason_155}"
                )
        else:
            print(f"  [CONVEYOR_SAFE_LIFT_155] no z raise: {conv_reason_155}")

        # 155_: APalt 위 정답지 큐브 center를 최종 place center로 사용한다.
        slot_idx_155 = int(getattr(task, "_stack_slot_index", 0))
        marker_center_155, marker_reason_155 = resolve_slot_marker_center_155(stage, slot_idx_155, fallback_center=task.goal_center)
        boxaprop_place_enabled = bool(VERTICAL_LIFT_THEN_CUBE_OVER_ENABLED)
        if marker_center_155 is not None:
            place_center = marker_center_155.copy()
            boxaprop_place_enabled = True
            print(
                f"  [SLOT_MARKER_GOAL_155] using marker as final box center: {marker_reason_155}, "
                f"place_center=({place_center[0]:.4f},{place_center[1]:.4f},{place_center[2]:.4f})"
            )
        else:
            place_center = np.array(task.goal_center, dtype=float).copy() if task.goal_center is not None else attach_center.copy()
            print(f"  [SLOT_MARKER_GOAL_155] fallback to computed goal_center; reason={marker_reason_155}")

        move_suction = lift_suction.copy()
        lower_suction = lift_suction.copy()
        if boxaprop_place_enabled:
            # place_center는 최종 박스 중심 좌표다. 박스 중심이 slot marker에 오도록 suction 목표를 역산한다.
            lower_suction = place_center - attached_center_offset
            move_suction = lower_suction.copy()
            move_suction[2] = max(
                float(lift_suction[2]),
                float(lower_suction[2]) + float(SLOT_MARKER_APPROACH_CLEARANCE_Z_155),
            )

        phase_sequence = [
            {"name": "vertical_lift", "kind": "path_lift", "start": attach_suction_pos.copy(), "target": lift_suction.copy(), "steps": int(TOP_LOCK_PATH_LIFT_STEPS)},
        ]

        if CUSTOM_CARRY_MODE == "VERTICAL_JOINT1_REVERSE":
            # 117_: 긴 Cartesian XY 이동 대신, 수직 상승 후 joint_1/link_1 하나만 회전한다.
            # joint_1은 로봇 베이스 기준 yaw라서 상자를 뒤쪽 큐브 방향으로 보내는 주 회전축이다.
            # reverse_vertical_lower의 start/target은 joint_1 회전이 끝난 시점의 실제 suction 위치로 동적 설정된다.
            phase_sequence.append({
                "name": "joint1_rotate_to_back",
                "kind": "joint1_rotate",
                "steps": int(HYBRID_JOINT1_ROTATE_STEPS),
                "mode": str(HYBRID_JOINT1_ROTATE_MODE),
                "sign": float(HYBRID_JOINT1_ROTATE_SIGN),
                "fallback_deg": float(get_stack_joint1_deg_145(getattr(task, "_stack_slot_index", 0))),
                "stack_slot_index_145": int(getattr(task, "_stack_slot_index", 0)),
                "max_deg": float(HYBRID_JOINT1_ROTATE_MAX_DEG),
            })
            if bool(globals().get("SLOT_MARKER_PALLETIZING_ENABLED_155", False)) and bool(boxaprop_place_enabled):
                # 155_: link_1 회전은 큰 방향 전환까지만 담당한다.
                # 최종 위치는 정답지 큐브 위로 짧게 보정한 뒤 하강한다.
                phase_sequence.append({
                    "name": "slot_marker_move_over_155",
                    "kind": "path_move",
                    "start": lift_suction.copy(),
                    "target": move_suction.copy(),
                    "steps": int(SLOT_MARKER_FINAL_MOVE_STEPS_155),
                })
                phase_sequence.append({
                    "name": "slot_marker_lower_155",
                    "kind": "path_lower",
                    "start": move_suction.copy(),
                    "target": lower_suction.copy(),
                    "steps": int(SLOT_MARKER_FINAL_LOWER_STEPS_155),
                    "fused_yaw_align_184": bool(globals().get("FUSED_SECOND_BOX_YAW_ALIGN_ENABLED_184", False)),
                })
                phase_sequence.append({
                    "name": "slot_marker_settle_155",
                    "kind": "path_settle",
                    "start": lower_suction.copy(),
                    "target": lower_suction.copy(),
                    "steps": int(SLOT_MARKER_FINAL_SETTLE_STEPS_155),
                    "fused_yaw_align_184": bool(globals().get("FUSED_SECOND_BOX_YAW_ALIGN_ENABLED_184", False)),
                })
                _insert_pre_release_yaw_184 = bool(globals().get("PRE_RELEASE_YAW_ALIGN_RMPFLOW_ENABLED_171", True)) and bool(globals().get("PRE_RELEASE_YAW_ALIGN_INSERT_AFTER_SETTLE_171", True))
                if (
                    _insert_pre_release_yaw_184
                    and bool(globals().get("FUSED_SECOND_BOX_YAW_ALIGN_ENABLED_184", False))
                    and bool(globals().get("FUSED_YAW_ALIGN_DISABLE_FINAL_PRE_RELEASE_PHASE_184", True))
                    and int(getattr(task, "_stack_slot_index", 0)) in tuple(int(x) for x in globals().get("FUSED_YAW_ALIGN_SLOT_INDICES_184", (1,)))
                ):
                    _insert_pre_release_yaw_184 = False
                    print(f"  [FUSED_YAW_ALIGN_185] slot_index={int(getattr(task, '_stack_slot_index', 0)) + 1} 상자는 slot_marker_lower/settle 중 yaw를 같이 맞추므로 별도 pre_release_yaw_align_171 phase를 생략합니다.")
                if _insert_pre_release_yaw_184:
                    phase_sequence.append({
                        "name": "pre_release_yaw_align_171",
                        "kind": "pre_release_yaw_align_171",
                        "steps": int(globals().get("PRE_RELEASE_YAW_ALIGN_STEPS_171", 70)),
                    })
                    print("  [PRE_RELEASE_YAW_ALIGN_173] phase inserted after slot_marker_settle_155 and before release - REAL RMPFlow action for full steps")
            else:
                # fallback: 154/147 방식. joint_1 회전 후 현재 위치에서 반대 방향으로 수직 하강.
                phase_sequence.append({
                    "name": "reverse_vertical_lower",
                    "kind": "reverse_vertical_lower",
                    "steps": int(HYBRID_REVERSE_LOWER_STEPS),
                    "delta_z": -float(VERTICAL_LIFT_DELTA_Z) + float(globals().get("DROP_RELEASE_EXTRA_Z_125", 0.0)),
                })
                phase_sequence.append({
                    "name": "hybrid_settle",
                    "kind": "path_settle",
                    "steps": int(HYBRID_SETTLE_STEPS),
                })
        elif boxaprop_place_enabled:
            phase_sequence.append({"name": "boxaprop_move", "kind": "path_move", "start": lift_suction.copy(), "target": move_suction.copy(), "steps": int(CUBE_OVER_MOVE_STEPS)})
            phase_sequence.append({"name": "boxaprop_lower", "kind": "path_lower", "start": move_suction.copy(), "target": lower_suction.copy(), "steps": int(BOXAPROP_LOWER_STEPS)})
            phase_sequence.append({"name": "boxaprop_settle", "kind": "path_settle", "start": lower_suction.copy(), "target": lower_suction.copy(), "steps": int(BOXAPROP_SETTLE_STEPS)})
        else:
            phase_sequence.append({"name": "vertical_hold", "kind": "path_settle", "start": lift_suction.copy(), "target": lift_suction.copy(), "steps": int(VERTICAL_LIFT_HOLD_STEPS)})

        rr_place = robot_relative_vector(stage, place_center)
        print(
            f"  [SIMPLE_3STEP_ROUTE_117] attach_suction=({attach_suction_pos[0]:.3f},{attach_suction_pos[1]:.3f},{attach_suction_pos[2]:.3f}), "
            f"lift_suction=({lift_suction[0]:.3f},{lift_suction[1]:.3f},{lift_suction[2]:.3f}), "
            f"delta_z={VERTICAL_LIFT_DELTA_Z:.3f}, drop_extra_z_125={float(globals().get('DROP_RELEASE_EXTRA_Z_125', 0.0)):.3f}, boxaprop_place_enabled={boxaprop_place_enabled}, "
            f"move_suction=({move_suction[0]:.3f},{move_suction[1]:.3f},{move_suction[2]:.3f}), "
            f"lower_suction=({lower_suction[0]:.3f},{lower_suction[1]:.3f},{lower_suction[2]:.3f}), "
            f"place_center_world=({place_center[0]:.4f}, {place_center[1]:.4f}, {place_center[2]:.4f}), "
            f"place_center_robot_relative=({rr_place[0]:+.4f}, {rr_place[1]:+.4f}, {rr_place[2]:+.4f}), "
            f"follow_actual_suction={not TOP_LOCK_FOLLOW_DESIRED_PATH}, custom_mode={CUSTOM_CARRY_MODE}, "
            f"stack_slot={int(getattr(task, '_stack_slot_index', 0)) + 1}, joint1_deg_145={get_stack_joint1_deg_145(getattr(task, '_stack_slot_index', 0)):.1f}"
        )
        return {
            "place_center": place_center.copy(),
            "safe_z": float(lift_suction[2]),
            "lift_suction": lift_suction,
            "move_suction": move_suction.copy(),
            "lower_suction": lower_suction.copy(),
            "phase_sequence": phase_sequence,
            "vertical_only": True,
            "boxaprop_place_enabled": boxaprop_place_enabled,
            "boxaprop_release": bool(HYBRID_RELEASE_ON_SETTLE) if CUSTOM_CARRY_MODE == "VERTICAL_JOINT1_REVERSE" else bool(BOXAPROP_RELEASE_ON_SETTLE),
            "cube_over_enabled": boxaprop_place_enabled,
            "cube_over_suction": move_suction.copy(),
            "cube_over_box_center": place_center.copy(),
        }

    place_center = np.array(task.goal_center, dtype=float).copy()
    # 56_: 2x2 적재에서는 task.goal_center의 Z가 BoxAprop 윗면/아래층 박스 높이를 반영한다.
    # 이전 방식처럼 pick 시점의 attach_center.z로 덮어쓰면 slot 높이가 무시되어 가지런히 쌓이지 않는다.
    if not MULTI_ORIBOX_STACKING_ENABLED:
        place_center[2] = float(attach_center[2])

    robot_center, _ = get_robot_center_for_goal(stage)
    # 49_: BoxAprop 위 3칸 적재에서는 목표가 BoxAprop 기준이어야 한다.
    # 기존처럼 로봇/발판 guard 밖으로 밀어내면 slot 위치가 변해서 첫 박스부터 엉뚱한 방향으로 간다.
    if robot_center is not None and not (MULTI_ORIBOX_STACKING_ENABLED and STACK_USE_DIRECT_CARRY_ROUTE):
        place_center[:2] = _push_xy_outside_robot_guard(stage, place_center[:2], center_xy=np.array(robot_center, dtype=float)[:2])

    safe_z = max(
        float(CUSTOM_CARRY_SAFE_SUCTION_Z),
        float(attach_suction_pos[2]) + float(CUSTOM_LIFT_DELTA_Z),
    )

    lift_suction = attach_suction_pos.copy()
    lift_suction[2] = safe_z

    move_suction = place_center - attached_center_offset
    move_suction[2] = safe_z

    lower_suction = place_center - attached_center_offset
    lower_suction[2] = max(float(lower_suction[2]) + float(CUSTOM_LOWER_CLEARANCE_Z), 0.10)

    if MULTI_ORIBOX_STACKING_ENABLED and STACK_USE_DIRECT_CARRY_ROUTE:
        # 49_: 적재는 단순하고 안정적인 경로로 처리한다.
        # lift -> BoxAprop slot 위로 수평 이동 -> slot 위로 하강
        # 59_: 각 phase의 목표를 한 번에 던지지 않고, 시작점->목표점을 고정 보간한다.
        # 중간에 target_suction을 재계산하지 않으므로 56_처럼 목표가 튀지 않는다.
        phase_sequence = [
            {"name": "lift", "kind": "path_lift", "start": attach_suction_pos.copy(), "target": lift_suction.copy(), "steps": int(TOP_LOCK_PATH_LIFT_STEPS)},
            {"name": "move_stack", "kind": "path_move", "start": lift_suction.copy(), "target": move_suction.copy(), "steps": int(TOP_LOCK_PATH_MOVE_STEPS)},
            {"name": "lower", "kind": "path_lower", "start": move_suction.copy(), "target": lower_suction.copy(), "steps": int(TOP_LOCK_PATH_LOWER_STEPS)},
            {"name": "settle", "kind": "path_settle", "start": lower_suction.copy(), "target": lower_suction.copy(), "steps": int(TOP_LOCK_PATH_SETTLE_STEPS)},
        ]
        print(
            f"  [STACK_DIRECT_ROUTE] start=({lift_suction[0]:.3f},{lift_suction[1]:.3f},{lift_suction[2]:.3f}), "
            f"move=({move_suction[0]:.3f},{move_suction[1]:.3f},{move_suction[2]:.3f}), "
            f"lower=({lower_suction[0]:.3f},{lower_suction[1]:.3f},{lower_suction[2]:.3f})"
        )
    else:
        phase_sequence = _make_avoidance_suction_waypoints(stage, lift_suction, move_suction)
        phase_sequence.append({"name": "lower", "kind": "lower", "target": lower_suction.copy()})

    return {
        "place_center": place_center,
        "safe_z": safe_z,
        "lift_suction": lift_suction,
        "move_suction": move_suction,
        "lower_suction": lower_suction,
        "phase_sequence": phase_sequence,
    }

def follow_box_to_suction(stage, task, suction_pos, attached_center_offset, min_center_z=None):
    """
    박스를 목표 좌표로 순간이동시키지 않는다.
    현재 suction point 위치 + attach 순간 offset만 따라가게 한다.
    """
    desired_box_center = _vec3(suction_pos) + _vec3(attached_center_offset)
    if min_center_z is not None:
        desired_box_center[2] = max(float(desired_box_center[2]), float(min_center_z))
    ok = move_box_center_to(
        stage,
        task.box_move_path,
        desired_center=desired_box_center,
        root_to_center_offset=task.box_root_to_center_offset,
    )
    zero_subtree_velocity(stage, task.box_move_path)
    return ok, desired_box_center


def snap_box_to_stack_slot_if_enabled(stage, task):
    """release 직전 slot 중심 강제 정렬. 50_에서는 기본 False라 순간이동하지 않는다."""
    if not (MULTI_ORIBOX_STACKING_ENABLED and STACK_SNAP_BOX_TO_SLOT_ON_RELEASE):
        return False
    if task is None or task.goal_center is None:
        return False
    bbox = get_world_bbox_info(stage, task.box_path)
    if bbox is None:
        return False
    current_center = np.array(bbox["center"], dtype=float)
    goal_center = np.array(task.goal_center, dtype=float)
    xy_error = float(np.linalg.norm(current_center[:2] - goal_center[:2]))
    if xy_error > float(STACK_SNAP_MAX_XY_ERROR):
        print(f"  [STACK_SNAP_SKIP] xy_error={xy_error:.3f} > {STACK_SNAP_MAX_XY_ERROR:.3f}, current={current_center}, goal={goal_center}")
        return False
    ok = move_box_center_to(
        stage,
        task.box_move_path,
        desired_center=goal_center,
        root_to_center_offset=task.box_root_to_center_offset,
    )
    zero_subtree_velocity(stage, task.box_move_path)
    print(
        f"  [STACK_SNAP] ok={ok}, xy_error_before={xy_error:.3f}, "
        f"goal_center=({goal_center[0]:.3f},{goal_center[1]:.3f},{goal_center[2]:.4f})"
    )
    return ok

# ╔══════════════════════════════════════════════════════════════╗
# ║  Main                                                       ║
# ╚══════════════════════════════════════════════════════════════╝
def initialize_robot_for_conveyor(robot, world):
    global ROBOT_INITIAL_JOINTS_156
    robot.initialize()
    if RESET_ROBOT_TO_ZERO:
        robot.set_joint_positions(np.zeros(robot.num_dof))
    try:
        ROBOT_INITIAL_JOINTS_156 = np.array(robot.get_joint_positions(), dtype=float).copy()
        if bool(globals().get("ABSOLUTE_JOINT1_LOG_156", False)):
            j1 = float(ROBOT_INITIAL_JOINTS_156[0]) if len(ROBOT_INITIAL_JOINTS_156) > 0 else 0.0
            print(f"[ABS_JOINT1_INIT_156] captured initial joints. initial_j1={j1:+.6f} rad/{math.degrees(j1):+.1f}deg")
    except Exception as exc:
        ROBOT_INITIAL_JOINTS_156 = None
        print(f"[ABS_JOINT1_INIT_156][WARN] failed to capture initial joints: {exc}")


def main():
    my_world = World(stage_units_in_meters=1.0)
    task = M0609ConveyorBoxTask(name="m0609_conveyor_oribox_task")
    my_world.add_task(task)
    my_world.reset()

    stage = omni.usd.get_context().get_stage()

    # reset/recompose 이후에도 기존 RG2가 다시 보이는 경우를 막기 위해 한 번 더 제거한다.
    remove_old_gripper(stage, verbose=True)
    # 109_: reset/recompose 후에도 사용자가 보낸 Cube가 viewport에 남아 있도록 다시 visible 처리한다.
    force_show_cube_prims_109(stage, verbose=True)
    # 146_: reset/recompose 이후에도 Environment 계열 prim을 다시 visible/active로 복구한다.
    force_show_environment_prims_146(stage, verbose=True)
    repair_robotaprop_and_oribox_colliders(stage, completed_roots=set(), verbose=True)

    robot = my_world.scene.get_object("m0609_robot")
    initialize_robot_for_conveyor(robot, my_world)

    for _ in range(30):
        update_vgc10_suction_anchor(robot)
        my_world.step(render=True)

    print("\n" + "=" * 60)
    print("[C-2] PickPlaceController 생성")
    print("=" * 60)
    print(f"  Conveyor USD = {USD_PATH}")
    print(f"  box path     = {task.box_path}")
    print(f"  move path    = {task.box_move_path}")
    print(f"  box center   = {task.box_initial_center}")
    print(f"  box top      = {task.box_initial_top_center}")
    print(f"  box height   = {task.box_height}")
    print(f"  pick target  = {task.pick_center}  # controller picking_position")
    print(f"  goal center  = {task.goal_center}")
    print(f"  URDF         = {M0609_URDF_PATH}")
    print(f"  description  = {M0609_DESCRIPTION_PATH}")
    print(f"  rmpflow      = {M0609_RMPFLOW_CONFIG_PATH}")
    print(f"  events_dt    = {EVENTS_DT}")
    print(f"  EE frame     = {EE_LINK_NAME}")
    print(f"  attach mode  = top-surface-rectangle, z_gap=[{BOX_ATTACH_Z_MIN},{BOX_ATTACH_Z_MAX}], dist<={BOX_ATTACH_DIST_TOL}")
    print(f"  stop gate    = path={STOP_CHECK_PRIM_PATH}, required_steps={BOX_STABLE_REQUIRED_STEPS}, move<={BOX_STABLE_POS_TOL}m/step")
    print(f"  axis diag    = enabled={AXIS_YAW_DIAGNOSTIC_ENABLED_164}, stop_after_release={DIAG_STOP_AFTER_RELEASE_COUNT_164 if DIAG_STOP_AFTER_RELEASE_ENABLED_164 else None}")
    print(f"  stop mode    = bbox_move={BOX_STOP_USE_BBOX_MOVE}, linear_vel_gate={BOX_STOP_USE_LINEAR_VEL}, angular_vel_gate={BOX_STOP_USE_ANGULAR_VEL}")
    print(f"  carry mode   = disable_physics_during_carry={BOX_DISABLE_PHYSICS_DURING_CARRY}, reenable_after_release={BOX_REENABLE_PHYSICS_AFTER_RELEASE}")
    print(f"  debug log    = every_step={BOX_STOP_LOG_EVERY_STEP}, interval={BOX_STOP_LOG_INTERVAL}, note=linear/angular velocity are INFO_ONLY unless gate=True")

    null_gripper = NullGripper()

    controller = PickPlaceController(
        name="m0609_pick_place_controller",
        gripper=null_gripper,
        robot_articulation=robot,
        end_effector_initial_height=0.30,
        events_dt=EVENTS_DT,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )
    cart_controller = RMPFlowController(
        name="m0609_custom_carry_controller",
        robot_articulation=robot,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )
    print("  [OK] PickPlaceController 생성 완료")
    print("  [OK] Custom RMPFlow carry controller 생성 완료")
    print("  [30_] infinite retry loop: joint lift -> base swing -> release -> home -> wait/retry")

    print("\n[VGC10 + Conveyor OriBox root suction 시작]\n")
    was_playing = False
    task_done = False
    attached = False
    ever_attached = False
    released = False
    returning_home = False
    home_start_joints = None
    home_target_joints = None
    home_return_step = 0
    retry_logged = False
    best_attach_reason = "none"
    best_attach_dist = 999.0

    # attach 순간의 bbox center offset. suction point를 따라 박스 bbox center를 움직일 때 사용한다.
    attached_center_offset = None
    attach_center_z = None
    attach_steps = 0

    custom_carry_active = False
    custom_carry_phase = None
    custom_phase_step = 0
    custom_targets = None
    custom_fixed_orientation = None
    custom_min_center_z = None
    custom_phase_index = 0
    release_watchdog_counter_127 = 0

    # 박스가 완전히 멈춘 뒤에만 로봇 pick-place를 시작한다.
    box_stop_detector = BoxStopDetector()
    pick_zone_detector = PickZoneDetector()
    box_stopped_for_pick = False
    pick_started = False

    # 53_: 작업영역 진입 후 흡착 전에 suction point를 box_top 좌표로 직접 정렬하는 상태값
    pre_attach_align_active = False
    pre_attach_align_phase = "idle"
    pre_attach_align_step = 0
    pre_attach_fixed_orientation = None

    # 30_: 무한 반복 상태값
    loop_cycle_index = 0
    loop_attempt_index = 0
    home_return_reason = "initial"
    home_return_is_success = False
    ignore_released_center = None
    ignore_wait_counter = 0

    # 47_: OriBoxA_01~03 다중 박스 쌓기 상태값
    completed_box_roots = set()
    zone_counts_by_root = {}
    stack_slot_index = 0
    stack_platform_lower_count = 0
    stack_complete_logged = False

    # ================================================================
    # 포크리프트 적재 시퀀스 설정  ← 여기 숫자만 바꾸면 됩니다
    # ================================================================
    # 145_: 이번 사이클에서 APalt에 실을 상자 개수.
    #       1,2,3,4 중 하나로 바꾸면 전체 흐름이 바뀐다.
    #       2 = OriBoxA_01, OriBoxA_02 release 후 카운트다운/운반
    #       3/4 = 2개 release 후 APalt를 BOX_STACK_HEIGHT_145만큼 내리고, 나머지를 쌓은 뒤 운반
    FORKLIFT_TRIGGER_COUNT  = 4  # 185_: 4개까지 연속 적재
    # 139_: release count가 조건에 도달해도 바로 forklift를 움직이지 않고 카운트다운 후 시작한다.
    FORKLIFT_START_DELAY_SEC_139 = 5
    # 139_: 평상시 main_loop forklift 위치 로그는 끈다. delay 중에는 5초 카운트만 출력한다.
    FORKLIFT_TRACK_MAIN_LOOP_139 = False
    # 144_: 화면/옆면 기준 좌표 표현으로 정리한다.
    #       - 코드상 X = 트럭이 앞으로 가는 방향 = USD GUI Translate Y
    #       - 코드상 Y = 리프트가 위로 올라가는 방향 = USD GUI Translate Z
    #       - USD GUI Translate X는 사용하지 않는다.
    FORKLIFT_LIFT_TARGET_Y  = 0.4    # 코드상 Y축 상승 높이. 실제로는 USD Translate Z를 움직인다.
    FORKLIFT_LIFT_TARGET_Z  = FORKLIFT_LIFT_TARGET_Y  # 기존 함수 호환용 이름. 실제 USD 축은 Z.
    FORKLIFT_MOVE_TARGET_X  = -5.0    # 코드상 X축 전진 거리. 실제로는 USD Translate Y를 움직인다.
    FORKLIFT_MOVE_TARGET_Y  = 0.0    # 144_: 코드상 Y는 상승에 사용하므로 평면 Y 이동은 기본 사용 안 함.
    FORKLIFT_LIFT_WAIT_SEC  = 3.0    # 리프트 올린 후 안정화 대기 시간 (초)
    FORKLIFT_MOVE_SPEED     = 0.5    # 코드상 X 이동 속도 (m/s). 실제 USD Translate Y 속도.
    FORKLIFT_PRIM_PATH      = "/World/APalt"  # 141_: lift_flat 대신 /World/APalt를 가상 지게차/팔레트로 사용
    FORKLIFT_JOINT_PATH     = ""  # 141_: 실제 lift joint 사용 안 함. /World/APalt Xform을 직접 이동

    # 137_: 현재 USD의 forklift는 drive joint가 아니라 /World/forklift/S_ForkliftFork Xform 구조다.
    #       lift drive가 없으면 fork Xform 자체의 local Z를 직접 올려서 리프트 동작을 테스트한다.
    FORKLIFT_DIRECT_FORK_LIFT_ENABLED_137 = True
    # 138_: 사용자가 Stage에서 확인한 실제 포크 lift prim.
    # 이 Xform을 local Z로 올리면 내부 node_/mesh_까지 같이 올라간다.
    FORKLIFT_DIRECT_FORK_PATHS_137 = (
        "/World/APalt",
    )

    # 137_: forklift trigger=1 테스트 중에도 stack_slot_count=1 때문에 로봇 루프가 멈추지 않게 한다.
    #       slot 계산은 기존 1번 slot을 쓰되, stack full 조건만 무시한다.
    ALLOW_CONTINUE_AFTER_STACK_FULL_137 = True
    # ================================================================

    forklift_sequence_triggered = False  # 시퀀스가 이미 실행됐는지 체크

    print("[VIRTUAL_APALT_TARGET_144] root=/World/APalt  (lift_flat/forklift 모델 사용 안 함)")
    print("[VIRTUAL_APALT_TARGET_144] side-view code axes: X=truck direction(USD Translate Y), Y=lift up(USD Translate Z)")

    # ================================================================
    # 136_: forklift 동작 확인용 위치 추적 + 안전 경로 자동 탐색
    # - 기존 FORKLIFT_JOINT_PATH가 USD와 다르면 null prim 오류가 났다.
    # - 이제 root / lift joint / targetPosition attribute를 stage에서 자동 탐색한다.
    # - 못 찾으면 오류로 멈추지 않고 후보 경로를 출력한 뒤 Y 이동 테스트만 계속한다.
    # ================================================================
    FORKLIFT_ROOT_CANDIDATE_PATHS_136 = (
        FORKLIFT_PRIM_PATH,
    )
    # 142_: /World/APalt 가상 지게차 모드에서는 실제 lift joint를 전혀 쓰지 않는다.
    # 빈 path("")나 예전 lift_flat 후보를 넣으면 Property/UI async 갱신 중 SdfPath 경고가 날 수 있어 제거한다.
    FORKLIFT_LIFT_JOINT_CANDIDATE_PATHS_136 = ()
    FORKLIFT_LIFT_TARGET_ATTR_NAMES_136 = (
        "drive:linear:physics:targetPosition",
        "drive:linear:targetPosition",
        "physics:targetPosition",
        "targetPosition",
    )

    FORKLIFT_POSE_TRACK_ENABLED = True
    FORKLIFT_POSE_LOG_INTERVAL = 20
    FORKLIFT_POSE_LOG_EVERY_STEP = False
    FORKLIFT_POSE_JUMP_WARN_TOL = 0.010
    _FORKLIFT_TRACK_STATE_135 = {"step": 0, "prev_root": None, "prev_center": None}
    _FORKLIFT_RESOLVE_CACHE_136 = {"root_path": None, "lift_path": None, "attr_name": None}

    # 141_: lift_flat/forklift 모델을 쓰지 않고 /World/APalt 하나를 가상 지게차/팔레트처럼 움직인다.
    # release 완료된 상자들은 APalt 이동 시작 직전에 kinematic 상태로 고정하고,
    # APalt bbox center 기준 상대 offset을 저장한 뒤 APalt 이동량을 그대로 따라가게 한다.
    VIRTUAL_APALT_FORKLIFT_ENABLED_141 = True
    VIRTUAL_APALT_PATH_141 = "/World/APalt"
    VIRTUAL_APALT_CARGO_ROOTS_141 = []
    VIRTUAL_APALT_CARGO_OFFSETS_141 = {}
    VIRTUAL_APALT_CARGO_DISABLE_COLLISION_141 = False  # True로 바꾸면 이동 중 상자 충돌도 꺼짐
    VIRTUAL_APALT_LOG_INTERVAL_141 = 30
    # 142_: 지게차처럼 천천히 상승하도록 APalt Z 이동을 step 단위로 수행한다.
    VIRTUAL_APALT_LIFT_SPEED_142 = 0.08  # m/s. 0.4m 상승이면 약 5초
    VIRTUAL_APALT_LIFT_LOG_INTERVAL_142 = 15

    def _resolve_forklift_root_path_136(stage_now, verbose=False):
        """USD 안의 forklift root prim을 안전하게 찾는다."""
        cached = _FORKLIFT_RESOLVE_CACHE_136.get("root_path")
        if cached:
            try:
                prim = stage_now.GetPrimAtPath(cached)
                if prim and prim.IsValid():
                    return str(cached)
            except Exception:
                pass

        for p in FORKLIFT_ROOT_CANDIDATE_PATHS_136:
            try:
                prim = stage_now.GetPrimAtPath(p)
                if prim and prim.IsValid():
                    _FORKLIFT_RESOLVE_CACHE_136["root_path"] = str(p)
                    if verbose:
                        print(f"[FORKLIFT_RESOLVE_136] root found by candidate: {p}")
                    return str(p)
            except Exception:
                pass

        found = []
        try:
            for prim in stage_now.Traverse():
                ps = prim.GetPath().pathString
                low = ps.lower()
                if "forklift" in low:
                    found.append(ps)
            if found:
                # 가장 짧은 path를 root 후보로 사용한다.
                found_sorted = sorted(found, key=lambda x: (x.count("/"), len(x)))
                root_path = found_sorted[0]
                _FORKLIFT_RESOLVE_CACHE_136["root_path"] = root_path
                if verbose:
                    print(f"[FORKLIFT_RESOLVE_136] root auto-selected: {root_path}")
                    print(f"[FORKLIFT_RESOLVE_136] forklift candidates sample={found_sorted[:12]}")
                return root_path
        except Exception as exc:
            if verbose:
                print(f"[FORKLIFT_RESOLVE_WARN_136] root scan failed: {exc}")

        if verbose:
            print(f"[FORKLIFT_RESOLVE_FAIL_136] forklift root not found. expected={FORKLIFT_PRIM_PATH}")
        return None

    def _get_lift_drive_attr_from_prim_136(prim):
        """prim에서 lift targetPosition 계열 attribute를 찾는다."""
        if prim is None:
            return None, None
        try:
            if not prim.IsValid():
                return None, None
        except Exception:
            return None, None

        for attr_name in FORKLIFT_LIFT_TARGET_ATTR_NAMES_136:
            try:
                attr = prim.GetAttribute(attr_name)
                if attr and attr.IsValid():
                    return attr, attr_name
            except Exception:
                pass

        try:
            for attr in prim.GetAttributes():
                name = attr.GetName()
                low = name.lower()
                if "targetposition" in low and ("drive" in low or "physics" in low or "linear" in low):
                    return attr, name
        except Exception:
            pass
        return None, None

    def _resolve_forklift_lift_drive_attr_136(stage_now, verbose=False):
        """lift joint prim과 targetPosition attr를 자동 탐색한다."""
        cached_path = _FORKLIFT_RESOLVE_CACHE_136.get("lift_path")
        cached_attr = _FORKLIFT_RESOLVE_CACHE_136.get("attr_name")
        if cached_path and cached_attr:
            try:
                prim = stage_now.GetPrimAtPath(cached_path)
                attr = prim.GetAttribute(cached_attr) if prim and prim.IsValid() else None
                if attr and attr.IsValid():
                    return prim, attr, str(cached_attr)
            except Exception:
                pass

        for p in FORKLIFT_LIFT_JOINT_CANDIDATE_PATHS_136:
            try:
                prim = stage_now.GetPrimAtPath(p)
                attr, attr_name = _get_lift_drive_attr_from_prim_136(prim)
                if attr is not None:
                    _FORKLIFT_RESOLVE_CACHE_136["lift_path"] = str(p)
                    _FORKLIFT_RESOLVE_CACHE_136["attr_name"] = str(attr_name)
                    if verbose:
                        print(f"[FORKLIFT_RESOLVE_136] lift drive found by candidate: prim={p}, attr={attr_name}")
                    return prim, attr, str(attr_name)
            except Exception:
                pass

        root_path = _resolve_forklift_root_path_136(stage_now, verbose=verbose)
        found_candidates = []
        try:
            for prim in stage_now.Traverse():
                ps = prim.GetPath().pathString
                low = ps.lower()
                if root_path and not ps.startswith(root_path):
                    continue
                if ("lift" not in low) and ("fork" not in low) and ("joint" not in low):
                    continue

                attr, attr_name = _get_lift_drive_attr_from_prim_136(prim)
                if attr is not None:
                    _FORKLIFT_RESOLVE_CACHE_136["lift_path"] = ps
                    _FORKLIFT_RESOLVE_CACHE_136["attr_name"] = str(attr_name)
                    if verbose:
                        print(f"[FORKLIFT_RESOLVE_136] lift drive auto-selected: prim={ps}, attr={attr_name}")
                    return prim, attr, str(attr_name)

                # 디버그 후보: lift/joint 이름은 있지만 target attr가 없는 prim
                try:
                    target_attrs = [a.GetName() for a in prim.GetAttributes() if "target" in a.GetName().lower() or "drive" in a.GetName().lower()]
                except Exception:
                    target_attrs = []
                if len(found_candidates) < 20:
                    found_candidates.append((ps, prim.GetTypeName(), target_attrs[:8]))
        except Exception as exc:
            if verbose:
                print(f"[FORKLIFT_RESOLVE_WARN_136] lift drive scan failed: {exc}")

        if verbose:
            print(f"[FORKLIFT_RESOLVE_FAIL_136] lift drive attr not found. old_joint_path={FORKLIFT_JOINT_PATH}")
            for ps, typ, attrs in found_candidates[:12]:
                print(f"  [FORKLIFT_LIFT_CANDIDATE_136] path={ps}, type={typ}, attrs={attrs}")
        return None, None, None

    def _get_or_add_translate_op_137(xformable):
        """Xformable에서 translate op를 찾거나 없으면 만든다."""
        translate_op = None
        current_pos = None
        try:
            for op in xformable.GetOrderedXformOps():
                if "translate" in op.GetOpName():
                    translate_op = op
                    current_pos = op.Get()
                    break
        except Exception:
            pass

        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
            current_pos = Gf.Vec3d(0.0, 0.0, 0.0)

        if current_pos is None:
            current_pos = Gf.Vec3d(0.0, 0.0, 0.0)
        return translate_op, current_pos

    def _resolve_forklift_direct_fork_path_137(stage_now, verbose=False):
        """drive joint가 없는 forklift에서 실제로 들어올릴 fork Xform을 찾는다."""
        cached = _FORKLIFT_RESOLVE_CACHE_136.get("direct_fork_path")
        if cached:
            try:
                prim = stage_now.GetPrimAtPath(cached)
                if prim and prim.IsValid():
                    return str(cached)
            except Exception:
                pass

        for p in FORKLIFT_DIRECT_FORK_PATHS_137:
            try:
                prim = stage_now.GetPrimAtPath(p)
                if prim and prim.IsValid() and prim.IsA(UsdGeom.Xformable):
                    _FORKLIFT_RESOLVE_CACHE_136["direct_fork_path"] = str(p)
                    if verbose:
                        print(f"[FORKLIFT_RESOLVE_137] direct fork found by candidate: {p}")
                    return str(p)
            except Exception:
                pass

        root_path = _resolve_forklift_root_path_136(stage_now, verbose=verbose)
        candidates = []
        try:
            for prim in stage_now.Traverse():
                ps = prim.GetPath().pathString
                if root_path and not ps.startswith(root_path):
                    continue
                low = ps.lower()
                if "fork" not in low:
                    continue
                try:
                    is_xformable = prim.IsA(UsdGeom.Xformable)
                except Exception:
                    is_xformable = False
                candidates.append((ps, prim.GetTypeName(), is_xformable))
                # Mesh child보다 Xform parent를 우선한다.
                if is_xformable and prim.GetTypeName() in ("Xform", "Scope"):
                    _FORKLIFT_RESOLVE_CACHE_136["direct_fork_path"] = ps
                    if verbose:
                        print(f"[FORKLIFT_RESOLVE_137] direct fork auto-selected: {ps}")
                    return ps
        except Exception as exc:
            if verbose:
                print(f"[FORKLIFT_RESOLVE_WARN_137] direct fork scan failed: {exc}")

        if verbose:
            print("[FORKLIFT_RESOLVE_FAIL_137] direct fork Xform not found.")
            for ps, typ, is_xformable in candidates[:12]:
                print(f"  [FORKLIFT_FORK_CANDIDATE_137] path={ps}, type={typ}, xformable={is_xformable}")
        return None

    def _get_forklift_direct_fork_local_z_137(stage_now):
        fork_path = _resolve_forklift_direct_fork_path_137(stage_now, verbose=False)
        if not fork_path:
            return None, None
        try:
            prim = stage_now.GetPrimAtPath(fork_path)
            xform = UsdGeom.Xformable(prim)
            _op, pos = _get_or_add_translate_op_137(xform)
            return fork_path, float(pos[2])
        except Exception:
            return fork_path, None

    def forklift_set_direct_fork_lift_137(z_pos):
        """drive attr가 없을 때 fork Xform local Z를 직접 올리는 fallback 리프트."""
        if not bool(FORKLIFT_DIRECT_FORK_LIFT_ENABLED_137):
            return False
        stage_now = omni.usd.get_context().get_stage()
        fork_path = _resolve_forklift_direct_fork_path_137(stage_now, verbose=True)
        if not fork_path:
            print("[FORKLIFT][DIRECT_LIFT_SKIP_137] fork Xform을 찾지 못했습니다.")
            return False
        try:
            prim = stage_now.GetPrimAtPath(fork_path)
            xform = UsdGeom.Xformable(prim)
            translate_op, current_pos = _get_or_add_translate_op_137(xform)
            start_z = float(current_pos[2])
            target_z = float(z_pos)
            translate_op.Set(Gf.Vec3d(float(current_pos[0]), float(current_pos[1]), target_z))
            print(f"[FORKLIFT][DIRECT_LIFT_137] fork={fork_path}, local_z={start_z:.4f} -> {target_z:.4f}")
            log_forklift_pose_135("after_direct_fork_lift_137", force=True)
            return True
        except Exception as exc:
            print(f"[FORKLIFT][DIRECT_LIFT_ERROR_137] {type(exc).__name__}: {exc}")
            return False

    def log_forklift_pose_135(label="forklift", force=False):
        if not bool(FORKLIFT_POSE_TRACK_ENABLED):
            return
        try:
            _FORKLIFT_TRACK_STATE_135["step"] = int(_FORKLIFT_TRACK_STATE_135.get("step", 0)) + 1
            step = int(_FORKLIFT_TRACK_STATE_135["step"])
            if (not force) and (not bool(FORKLIFT_POSE_LOG_EVERY_STEP)) and (step % int(FORKLIFT_POSE_LOG_INTERVAL) != 0):
                return

            stage_now = omni.usd.get_context().get_stage()
            root_path = _resolve_forklift_root_path_136(stage_now, verbose=force)
            if not root_path:
                print(f"[FORKLIFT_TRACK_136][{label}][step={step}] MISSING forklift root. expected={FORKLIFT_PRIM_PATH}")
                return

            root_t = get_world_translation(stage_now, root_path)
            root_t = np.array(root_t if root_t is not None else [np.nan, np.nan, np.nan], dtype=float)
            bbox = get_world_bbox_info(stage_now, root_path)
            if bbox is not None:
                center = np.array(bbox.get("center", [np.nan, np.nan, np.nan]), dtype=float)
                top = np.array(bbox.get("top_center", [np.nan, np.nan, np.nan]), dtype=float)
            else:
                center = np.array([np.nan, np.nan, np.nan], dtype=float)
                top = np.array([np.nan, np.nan, np.nan], dtype=float)

            prev_root = _FORKLIFT_TRACK_STATE_135.get("prev_root")
            prev_center = _FORKLIFT_TRACK_STATE_135.get("prev_center")
            d_root = 0.0
            d_center = 0.0
            jump = ""
            if prev_root is not None:
                try:
                    d_root = float(np.linalg.norm(root_t - prev_root))
                    d_center = float(np.linalg.norm(center - prev_center))
                    if max(d_root, d_center) >= float(FORKLIFT_POSE_JUMP_WARN_TOL):
                        jump = f" MOVE>= {FORKLIFT_POSE_JUMP_WARN_TOL:.3f}"
                except Exception:
                    pass
            _FORKLIFT_TRACK_STATE_135["prev_root"] = root_t.copy()
            _FORKLIFT_TRACK_STATE_135["prev_center"] = center.copy()

            lift_target = None
            lift_path_txt = "None"
            try:
                prim, attr, attr_name = _resolve_forklift_lift_drive_attr_136(stage_now, verbose=False)
                if attr is not None:
                    lift_target = attr.Get()
                    lift_path_txt = prim.GetPath().pathString
            except Exception:
                lift_target = None

            # 137_: drive attr가 없는 forklift면 fork Xform local Z를 lift 값처럼 추적한다.
            if lift_target is None:
                try:
                    fork_path_137, fork_z_137 = _get_forklift_direct_fork_local_z_137(stage_now)
                    if fork_z_137 is not None:
                        lift_target = float(fork_z_137)
                        lift_path_txt = str(fork_path_137)
                except Exception:
                    pass

            lift_txt = "None" if lift_target is None else f"{float(lift_target):+.4f}"
            print(
                f"[FORKLIFT_TRACK_136][{label}][step={step}] "
                f"root_path={root_path} "
                f"root=({root_t[0]:+.4f},{root_t[1]:+.4f},{root_t[2]:+.4f}) "
                f"bbox_center=({center[0]:+.4f},{center[1]:+.4f},{center[2]:+.4f}) "
                f"bbox_top=({top[0]:+.4f},{top[1]:+.4f},{top[2]:+.4f}) "
                f"d_root={d_root:.4f} d_center={d_center:.4f}{jump} "
                f"lift_target={lift_txt} lift_prim={lift_path_txt}"
            )
        except Exception as exc:
            print(f"[FORKLIFT_TRACK_WARN_136][{label}] {exc}")

    def _virtual_apalt_get_center_141(stage_now):
        """APalt의 bbox center를 반환한다. bbox가 안 잡히면 prim world translation을 fallback으로 사용한다."""
        path = str(VIRTUAL_APALT_PATH_141)
        bbox = get_world_bbox_info(stage_now, path)
        if bbox is not None:
            try:
                return np.array(bbox.get("center"), dtype=float)
            except Exception:
                pass
        p = get_world_translation(stage_now, path)
        if p is None:
            return None
        return np.array(p, dtype=float)

    def _virtual_apalt_set_cargo_physics_frozen_141(stage_now, box_root_path, frozen=True):
        """APalt 이동 중 상자가 물리 반응하지 않도록 kinematic/gravity/velocity를 정리한다."""
        try:
            if frozen:
                set_prim_kinematic(stage_now, box_root_path, True)
                try:
                    rb = UsdPhysics.RigidBodyAPI.Apply(stage_now.GetPrimAtPath(box_root_path))
                    # kinematic body로 두고 중력만 꺼서 코드 위치 추종이 흔들리지 않게 한다.
                    if hasattr(rb, "CreateKinematicEnabledAttr"):
                        rb.CreateKinematicEnabledAttr(True).Set(True)
                    if hasattr(rb, "CreateRigidBodyEnabledAttr"):
                        rb.CreateRigidBodyEnabledAttr(True).Set(True)
                except Exception:
                    pass
                try:
                    from pxr import PhysxSchema
                    prim = stage_now.GetPrimAtPath(box_root_path)
                    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    physx_rb.CreateDisableGravityAttr(True).Set(True)
                except Exception:
                    pass
                zero_subtree_velocity(stage_now, box_root_path)
                if bool(VIRTUAL_APALT_CARGO_DISABLE_COLLISION_141):
                    set_subtree_collision_enabled(stage_now, box_root_path, False)
            else:
                if bool(VIRTUAL_APALT_CARGO_DISABLE_COLLISION_141):
                    set_subtree_collision_enabled(stage_now, box_root_path, True)
                set_prim_kinematic(stage_now, box_root_path, False)
        except Exception as exc:
            print(f"[VIRTUAL_APALT_CARGO_FREEZE_WARN_141] root={box_root_path}, frozen={frozen}, err={type(exc).__name__}: {exc}")

    def _virtual_apalt_freeze_apalt_physics_145(stage_now, frozen=True):
        """최종 운반 모드에서 APalt 자체도 물리 영향 없이 Xform으로만 움직이게 정리한다."""
        apalt_path = str(VIRTUAL_APALT_PATH_141)
        try:
            prim = stage_now.GetPrimAtPath(apalt_path)
            if not prim or not prim.IsValid():
                print(f"[VIRTUAL_APALT_FREEZE_SKIP_145] APalt prim 없음: {apalt_path}")
                return False
            if frozen:
                try:
                    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
                    if hasattr(rb, "CreateRigidBodyEnabledAttr"):
                        rb.CreateRigidBodyEnabledAttr(True).Set(True)
                    if hasattr(rb, "CreateKinematicEnabledAttr"):
                        rb.CreateKinematicEnabledAttr(True).Set(True)
                except Exception:
                    pass
                try:
                    from pxr import PhysxSchema
                    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    physx_rb.CreateDisableGravityAttr(True).Set(True)
                except Exception:
                    pass
                zero_subtree_velocity(stage_now, apalt_path)
            print(f"[VIRTUAL_APALT_FREEZE_145] apalt={apalt_path}, frozen={bool(frozen)}")
            return True
        except Exception as exc:
            print(f"[VIRTUAL_APALT_FREEZE_WARN_145] frozen={frozen}, err={type(exc).__name__}: {exc}")
            return False

    def _virtual_apalt_prepare_cargo_follow_141(stage_now):
        """completed_box_roots 안의 상자들을 APalt 기준 상대 offset으로 저장한다."""
        nonlocal VIRTUAL_APALT_CARGO_ROOTS_141, VIRTUAL_APALT_CARGO_OFFSETS_141
        apalt_center = _virtual_apalt_get_center_141(stage_now)
        if apalt_center is None:
            print(f"[VIRTUAL_APALT_PREP_FAIL_141] APalt center를 읽지 못했습니다: {VIRTUAL_APALT_PATH_141}")
            VIRTUAL_APALT_CARGO_ROOTS_141 = []
            VIRTUAL_APALT_CARGO_OFFSETS_141 = {}
            return False

        _virtual_apalt_freeze_apalt_physics_145(stage_now, True)

        roots = []
        offsets = {}
        for root in sorted(str(x) for x in completed_box_roots):
            prim = stage_now.GetPrimAtPath(root)
            if not prim or not prim.IsValid():
                continue
            box_pos = get_world_translation(stage_now, root)
            if box_pos is None:
                bbox = get_world_bbox_info(stage_now, root)
                if bbox is None:
                    continue
                box_pos = bbox.get("center")
            box_pos = np.array(box_pos, dtype=float)
            offsets[root] = box_pos - apalt_center
            roots.append(root)
            _virtual_apalt_set_cargo_physics_frozen_141(stage_now, root, True)
            print(
                f"[VIRTUAL_APALT_CARGO_LOCK_141] root={root}, "
                f"box=({box_pos[0]:+.4f},{box_pos[1]:+.4f},{box_pos[2]:+.4f}), "
                f"apalt_center=({apalt_center[0]:+.4f},{apalt_center[1]:+.4f},{apalt_center[2]:+.4f}), "
                f"offset=({offsets[root][0]:+.4f},{offsets[root][1]:+.4f},{offsets[root][2]:+.4f})"
            )

        VIRTUAL_APALT_CARGO_ROOTS_141 = roots
        VIRTUAL_APALT_CARGO_OFFSETS_141 = offsets
        print(f"[VIRTUAL_APALT_PREP_141] cargo_count={len(roots)}, roots={roots}")
        return True

    def _virtual_apalt_update_cargo_follow_141(stage_now, label="follow", force=False):
        """APalt 현재 center + 저장된 offset으로 cargo 상자들을 계속 따라가게 한다."""
        apalt_center = _virtual_apalt_get_center_141(stage_now)
        if apalt_center is None:
            return False
        for root in list(VIRTUAL_APALT_CARGO_ROOTS_141):
            offset = VIRTUAL_APALT_CARGO_OFFSETS_141.get(root)
            if offset is None:
                continue
            target = apalt_center + np.array(offset, dtype=float)
            set_prim_world_translation(stage_now, root, target)
            zero_subtree_velocity(stage_now, root)
        if force:
            print(
                f"[VIRTUAL_APALT_FOLLOW_141][{label}] apalt_center=({apalt_center[0]:+.4f},{apalt_center[1]:+.4f},{apalt_center[2]:+.4f}), "
                f"cargo_count={len(VIRTUAL_APALT_CARGO_ROOTS_141)}"
            )
        return True

    async def forklift_lift_smooth_142(z_offset, speed=None):
        """142_: /World/APalt를 지게차 리프트처럼 천천히 Z 방향으로 상승시킨다.
        cargo 상자들은 APalt 중심 기준 offset을 유지하며 같이 따라간다.
        simulation_app.update()를 쓰지 않고 next_update_async()만 사용해서 asyncio 재진입 오류를 피한다.
        """
        import omni.kit.app
        stage_now = omni.usd.get_context().get_stage()
        apalt_path = str(VIRTUAL_APALT_PATH_141)
        prim = stage_now.GetPrimAtPath(apalt_path)
        if not prim or not prim.IsValid():
            print(f"[VIRTUAL_APALT_LIFT_SKIP_142] APalt prim을 찾지 못했습니다: {apalt_path}")
            return False

        root_pos = get_world_translation(stage_now, apalt_path)
        center_before = _virtual_apalt_get_center_141(stage_now)
        if root_pos is None or center_before is None:
            print(f"[VIRTUAL_APALT_LIFT_SKIP_142] APalt 위치를 읽지 못했습니다: {apalt_path}")
            return False

        root_pos = np.array(root_pos, dtype=float)
        start_z = float(root_pos[2])
        target_z = start_z + float(z_offset)
        direction = 1.0 if float(z_offset) >= 0 else -1.0
        speed = float(VIRTUAL_APALT_LIFT_SPEED_142 if speed is None else speed)
        dt = 1.0 / 60.0
        step = max(1e-6, abs(speed) * dt) * direction
        current_z = start_z
        lift_i = 0

        print(
            f"[VIRTUAL_APALT_LIFT_START_142] apalt={apalt_path}, "
            f"start_z={start_z:+.4f}, target_z={target_z:+.4f}, dz={float(z_offset):+.4f}, speed={speed:.3f}m/s"
        )
        _virtual_apalt_update_cargo_follow_141(stage_now, label="lift_start_142", force=True)

        while abs(current_z - target_z) > abs(step):
            current_z += step
            root_now = root_pos.copy()
            root_now[2] = current_z
            set_prim_world_translation(stage_now, apalt_path, root_now)
            _virtual_apalt_update_cargo_follow_141(stage_now, label=f"lift:{lift_i}", force=False)
            lift_i += 1
            if lift_i % int(max(1, VIRTUAL_APALT_LIFT_LOG_INTERVAL_142)) == 0:
                _virtual_apalt_update_cargo_follow_141(stage_now, label=f"lift:{lift_i}", force=True)
                print(f"[VIRTUAL_APALT_LIFT_142] step={lift_i}, z={current_z:+.4f}/{target_z:+.4f}")
            await omni.kit.app.get_app().next_update_async()

        root_final = root_pos.copy()
        root_final[2] = target_z
        set_prim_world_translation(stage_now, apalt_path, root_final)
        _virtual_apalt_update_cargo_follow_141(stage_now, label="lift_done_142", force=True)
        center_after = _virtual_apalt_get_center_141(stage_now)
        print(
            f"[VIRTUAL_APALT_LIFT_DONE_142] apalt={apalt_path}, steps={lift_i}, "
            f"center_before=({center_before[0]:+.4f},{center_before[1]:+.4f},{center_before[2]:+.4f}), "
            f"center_after=({center_after[0]:+.4f},{center_after[1]:+.4f},{center_after[2]:+.4f})"
        )
        return True

    async def forklift_move_axis_smooth_143(axis_index, target_offset, speed=0.5, axis_name="Y"):
        """143_: /World/APalt를 지정 축(X/Y)으로 천천히 이동시킨다.
        cargo 상자들은 APalt 중심 기준 offset을 유지하며 같이 따라간다.
        axis_index: 0=X, 1=Y, 2=Z
        """
        import omni.kit.app
        stage_now = omni.usd.get_context().get_stage()
        apalt_path = str(VIRTUAL_APALT_PATH_141)
        prim = stage_now.GetPrimAtPath(apalt_path)
        if not prim or not prim.IsValid():
            print(f"[VIRTUAL_APALT_MOVE_SKIP_143] APalt prim을 찾지 못했습니다: {apalt_path}")
            return False

        target_offset = float(target_offset)
        if abs(target_offset) < 1e-9:
            print(f"[VIRTUAL_APALT_MOVE_SKIP_143] {axis_name} 이동 거리=0.0 → 건너뜀")
            _virtual_apalt_update_cargo_follow_141(stage_now, label=f"move_{axis_name.lower()}_skip_143", force=True)
            return True

        root_pos = get_world_translation(stage_now, apalt_path)
        if root_pos is None:
            print(f"[VIRTUAL_APALT_MOVE_SKIP_143] APalt 위치를 읽지 못했습니다: {apalt_path}")
            return False
        root_pos = np.array(root_pos, dtype=float)
        start_v = float(root_pos[int(axis_index)])
        target_v = start_v + target_offset
        direction = 1.0 if target_offset > 0 else -1.0
        dt = 1.0 / 60.0
        step = max(1e-6, abs(float(speed)) * dt) * direction
        current_v = start_v
        move_i = 0

        _virtual_apalt_update_cargo_follow_141(stage_now, label=f"move_{axis_name.lower()}_start_143", force=True)
        print(
            f"[VIRTUAL_APALT_MOVE_{axis_name}_START_143] "
            f"start_{axis_name.lower()}={start_v:+.4f}, target_{axis_name.lower()}={target_v:+.4f}, "
            f"offset={target_offset:+.4f}, speed={float(speed):.3f}"
        )

        while abs(current_v - target_v) > abs(step):
            current_v += step
            root_now = root_pos.copy()
            root_now[int(axis_index)] = current_v
            set_prim_world_translation(stage_now, apalt_path, root_now)
            _virtual_apalt_update_cargo_follow_141(stage_now, label=f"move_{axis_name.lower()}:{move_i}", force=False)
            move_i += 1
            if move_i % int(max(1, VIRTUAL_APALT_LOG_INTERVAL_141)) == 0:
                _virtual_apalt_update_cargo_follow_141(stage_now, label=f"move_{axis_name.lower()}:{move_i}", force=True)
                print(f"[VIRTUAL_APALT_MOVE_{axis_name}_143] step={move_i}, {axis_name.lower()}={current_v:+.4f}/{target_v:+.4f}")
            await omni.kit.app.get_app().next_update_async()

        root_final = root_pos.copy()
        root_final[int(axis_index)] = target_v
        set_prim_world_translation(stage_now, apalt_path, root_final)
        _virtual_apalt_update_cargo_follow_141(stage_now, label=f"move_{axis_name.lower()}_done_143", force=True)
        print(
            f"[VIRTUAL_APALT_MOVE_{axis_name}_DONE_143] "
            f"apalt={apalt_path}, {axis_name} 이동 완료: {start_v:.3f} -> {target_v:.3f}"
        )
        return True

    async def forklift_move_x_smooth_143(target_x_offset, speed=0.5):
        """144_: 코드상 X축 이동 = 트럭 방향 전진/후진.
        사용자가 GUI에서 확인한 실제 축은 USD Translate Y이므로 axis_index=1을 움직인다.
        cargo 상자들은 APalt 중심 offset을 유지하며 같이 따라간다.
        """
        return await forklift_move_axis_smooth_143(1, target_x_offset, speed=speed, axis_name="X")

    async def forklift_move_y_smooth(target_y_offset, speed=0.5):
        """141_: /World/APalt를 Y로 이동시키고, cargo 상자들은 APalt 중심 offset 기준으로 따라가게 한다."""
        import omni.kit.app
        stage_now = omni.usd.get_context().get_stage()
        apalt_path = str(VIRTUAL_APALT_PATH_141)
        prim = stage_now.GetPrimAtPath(apalt_path)
        if not prim or not prim.IsValid():
            print(f"[VIRTUAL_APALT_MOVE_SKIP_141] APalt prim을 찾지 못했습니다: {apalt_path}")
            return False
        root_pos = get_world_translation(stage_now, apalt_path)
        if root_pos is None:
            print(f"[VIRTUAL_APALT_MOVE_SKIP_141] APalt 위치를 읽지 못했습니다: {apalt_path}")
            return False
        root_pos = np.array(root_pos, dtype=float)
        start_y = float(root_pos[1])
        target_y = start_y + float(target_y_offset)
        direction = 1.0 if target_y_offset > 0 else -1.0
        dt = 1.0 / 60.0
        step = max(1e-6, abs(float(speed)) * dt) * direction
        current_y = start_y
        move_i = 0
        _virtual_apalt_update_cargo_follow_141(stage_now, label="move_start", force=True)
        print(f"[VIRTUAL_APALT_MOVE_141] start_y={start_y:+.4f}, target_y={target_y:+.4f}, speed={float(speed):.3f}")

        while abs(current_y - target_y) > abs(step):
            current_y += step
            root_now = root_pos.copy()
            root_now[1] = current_y
            set_prim_world_translation(stage_now, apalt_path, root_now)
            _virtual_apalt_update_cargo_follow_141(stage_now, label=f"move:{move_i}", force=False)
            move_i += 1
            if move_i % int(max(1, VIRTUAL_APALT_LOG_INTERVAL_141)) == 0:
                _virtual_apalt_update_cargo_follow_141(stage_now, label=f"move:{move_i}", force=True)
            await omni.kit.app.get_app().next_update_async()

        root_final = root_pos.copy()
        root_final[1] = target_y
        set_prim_world_translation(stage_now, apalt_path, root_final)
        _virtual_apalt_update_cargo_follow_141(stage_now, label="move_done", force=True)
        print(f"[VIRTUAL_APALT_MOVE_DONE_141] apalt={apalt_path}, Y 이동 완료: {start_y:.3f} -> {target_y:.3f}")
        return True

    async def forklift_load_sequence():
        """144_: 코드상 Y축 리프트업 → 코드상 X축 전진 시퀀스.
        실제 USD 축 매핑은 Y축 리프트업=Translate Z, X축 전진=Translate Y 이다.
        예외가 나도 asyncio task가 죽지 않게 막는다.
        """
        import asyncio as _asyncio
        try:
            print(f"[VIRTUAL_APALT] ===== 가상 APalt 지게차 시퀀스 시작 (A박스 {FORKLIFT_TRIGGER_COUNT}개 완료) =====")
            stage_now = omni.usd.get_context().get_stage()
            _virtual_apalt_prepare_cargo_follow_141(stage_now)
            _virtual_apalt_update_cargo_follow_141(stage_now, label="sequence_start", force=True)

            # 1단계: 코드상 Y축 상승. 실제 USD에서는 Translate Z가 변한다.
            extra_lift_145 = float(stack_platform_lower_count) * float(BOX_STACK_HEIGHT_145)
            final_lift_z_145 = float(FORKLIFT_LIFT_TARGET_Z) + extra_lift_145
            print(f"[VIRTUAL_APALT] 1단계: APalt 코드상 Y축 천천히 상승 → base_y={FORKLIFT_LIFT_TARGET_Y}, lowered_layers={stack_platform_lower_count}, extra_y={extra_lift_145:.4f}, final_y={final_lift_z_145:.4f}, actual_USD_Z, speed={VIRTUAL_APALT_LIFT_SPEED_142}m/s")
            lift_ok_136 = await forklift_lift_smooth_142(final_lift_z_145, speed=VIRTUAL_APALT_LIFT_SPEED_142)
            if lift_ok_136:
                # 상승 완료 후 짧게 안정화. sync update 사용 금지.
                await _asyncio.sleep(float(FORKLIFT_LIFT_WAIT_SEC))
                _virtual_apalt_update_cargo_follow_141(stage_now, label="after_lift_wait_142", force=True)
            else:
                await _asyncio.sleep(0.2)
                print("[VIRTUAL_APALT][LIFT_WARN_142] lift는 건너뛰고 Y 이동 테스트로 진행합니다.")

            # 2단계: 코드상 X축 전진. 실제 USD에서는 Translate Y가 변한다.
            print(f"[VIRTUAL_APALT] 2단계: APalt 코드상 X축 전진 → x={FORKLIFT_MOVE_TARGET_X}m (actual_USD_Y, 속도={FORKLIFT_MOVE_SPEED}m/s)")
            move_x_ok_143 = await forklift_move_x_smooth_143(FORKLIFT_MOVE_TARGET_X, speed=FORKLIFT_MOVE_SPEED)

            # 3단계: 144번에서는 평면 Y 이동을 사용하지 않는다. 코드상 Y는 리프트 높이로만 사용한다.
            move_y_ok_136 = True

            _virtual_apalt_update_cargo_follow_141(stage_now, label="sequence_done", force=True)
            print(
                f"[VIRTUAL_APALT] ===== 시퀀스 완료: "
                f"lift_ok={lift_ok_136}, move_x_ok={move_x_ok_143}, move_y_ok={move_y_ok_136}, "
                f"cargo={VIRTUAL_APALT_CARGO_ROOTS_141} ====="
            )
        except Exception as exc:
            print(f"[FORKLIFT_SEQUENCE_ERROR_136] {type(exc).__name__}: {exc}")
            try:
                log_forklift_pose_135("sequence_error", force=True)
            except Exception:
                pass
    import asyncio as _asyncio_module

    async def forklift_delayed_start_sequence_139(release_count):
        """139_: release count가 올라간 뒤 지정 시간 카운트다운만 출력하고 forklift sequence를 시작한다."""
        try:
            delay_sec = int(max(0, FORKLIFT_START_DELAY_SEC_139))
            for remain in range(delay_sec, 0, -1):
                print(f"[FORKLIFT_DELAY_139] release_count={release_count}/{FORKLIFT_TRIGGER_COUNT}, start_after={remain}s")
                await _asyncio_module.sleep(1.0)
            await forklift_load_sequence()
        except Exception as exc:
            print(f"[FORKLIFT_DELAY_ERROR_139] {type(exc).__name__}: {exc}")

    def check_and_trigger_forklift():
        """release count가 FORKLIFT_TRIGGER_COUNT에 도달하면 5초 카운트다운 후 forklift sequence 실행."""
        nonlocal forklift_sequence_triggered
        release_count = len(completed_box_roots)
        if (
            not forklift_sequence_triggered
            and release_count >= FORKLIFT_TRIGGER_COUNT
        ):
            forklift_sequence_triggered = True
            _asyncio_module.ensure_future(forklift_delayed_start_sequence_139(release_count))

    def _format_pose_145(v):
        if v is None:
            return "None"
        try:
            a = np.array(v, dtype=float)
            return f"({a[0]:+.4f},{a[1]:+.4f},{a[2]:+.4f})"
        except Exception:
            return str(v)

    def log_first_two_box_pose_145(stage_now, label, baseline=None, force=False):
        """APalt lowering 중 1번/2번 박스 위치 확인용 로그."""
        ap = _virtual_apalt_get_center_141(stage_now)
        msg = [f"[APALT_LOWER_BOX_TRACK_145][{label}] apalt_center={_format_pose_145(ap)}"]
        for root in ("/World/OriBoxA_01", "/World/OriBoxA_02"):
            pos = get_world_translation(stage_now, root)
            if pos is None:
                bbox = get_world_bbox_info(stage_now, root)
                pos = bbox.get("center") if bbox is not None else None
            delta_txt = ""
            if baseline is not None and root in baseline and pos is not None:
                d = np.array(pos, dtype=float) - np.array(baseline[root], dtype=float)
                delta_txt = f", d=({d[0]:+.4f},{d[1]:+.4f},{d[2]:+.4f})"
            msg.append(f"{root}={_format_pose_145(pos)}{delta_txt}")
        print(" | ".join(msg))

    def slow_lower_apalt_after_first_layer_145(stage_now, box_height=0.24):
        """3개 이상 적재할 때, 1층 2개 후 APalt를 상자 높이만큼 천천히 내린다.
        이때 1번/2번 박스 좌표를 로그로 계속 찍어서 튐/밀림 여부를 확인한다.
        """
        apalt_path = str(VIRTUAL_APALT_PATH_141)
        cur = get_world_translation(stage_now, apalt_path)
        if cur is None:
            cur_center = _virtual_apalt_get_center_141(stage_now)
            if cur_center is None:
                print(f"[APALT_LOWER_SKIP_145] APalt 위치를 읽지 못했습니다: {apalt_path}")
                return False
            cur = cur_center
        cur = np.array(cur, dtype=float)
        dz = float(max(0.01, box_height)) + float(STACK_LOWER_EXTRA_Z)
        steps = int(max(1, STACK_LOWER_SUPPORT_STEPS))
        target = cur.copy()
        target[2] -= dz

        baseline = {}
        for root in ("/World/OriBoxA_01", "/World/OriBoxA_02"):
            pos = get_world_translation(stage_now, root)
            if pos is None:
                bbox = get_world_bbox_info(stage_now, root)
                pos = bbox.get("center") if bbox is not None else None
            if pos is not None:
                baseline[root] = np.array(pos, dtype=float)

        print(
            f"[APALT_LOWER_START_145] trigger_count={FORKLIFT_TRIGGER_COUNT}, dz={dz:.4f}, steps={steps}, "
            f"apalt_start={_format_pose_145(cur)}, apalt_target={_format_pose_145(target)}, note={APALT_LOWER_SPEED_NOTE_145}"
        )
        log_first_two_box_pose_145(stage_now, "start", baseline=baseline, force=True)

        for i in range(1, steps + 1):
            alpha = float(i) / float(steps)
            now = cur * (1.0 - alpha) + target * alpha
            set_prim_world_translation(stage_now, apalt_path, now)
            if (i % int(max(1, APALT_LOWER_LOG_INTERVAL_145)) == 0) or i == 1 or i == steps:
                log_first_two_box_pose_145(stage_now, f"step={i}/{steps}", baseline=baseline, force=True)
            my_world.step(render=True)
            time.sleep(0.005)

        set_prim_world_translation(stage_now, apalt_path, target)
        log_first_two_box_pose_145(stage_now, "done", baseline=baseline, force=True)
        print(f"[APALT_LOWER_DONE_145] APalt lowered by box_height. dz={dz:.4f}, final={_format_pose_145(target)}")
        return True



    def wait_after_release_before_apalt_lower_160(stage_now, label="after_release"):
        """160_: release 직후 APalt를 바로 움직이지 않고 1~2초 정도 물리 안정화 시간을 준다."""
        if not bool(globals().get("POST_RELEASE_SETTLE_BEFORE_APALT_LOWER_ENABLED_160", True)):
            return
        steps_wait = int(max(0, globals().get("POST_RELEASE_SETTLE_BEFORE_APALT_LOWER_STEPS_160", 120)))
        log_interval = int(max(1, globals().get("POST_RELEASE_SETTLE_LOG_INTERVAL_160", 30)))
        if steps_wait <= 0:
            return
        print(f"[POST_RELEASE_SETTLE_160] start label={label}, steps={steps_wait}. APalt lower delayed until boxes settle physically.")
        for i in range(1, steps_wait + 1):
            my_world.step(render=True)
            if i == 1 or i == steps_wait or (i % log_interval) == 0:
                log_first_two_box_pose_145(stage_now, f"post_release_wait={i}/{steps_wait}", baseline=None, force=True)
            time.sleep(0.005)
        print(f"[POST_RELEASE_SETTLE_160] done label={label}, steps={steps_wait}")

    def handle_stack_release_count_145(stage_now, task, label="release"):
        """release 이후 중앙 통제 함수.
        - completed_box_roots에 root 등록
        - stack_slot_index 증가
        - FORKLIFT_TRIGGER_COUNT 기준으로 APalt lowering / 최종 운반 시작 통제
        """
        nonlocal stack_slot_index, stack_platform_lower_count
        root = get_task_completed_root_path(task)
        completed_box_roots.add(root)
        release_count = len(completed_box_roots)
        print(
            f"[RELEASE_COUNT_145][{label}] release_count={release_count}/{FORKLIFT_TRIGGER_COUNT}, "
            f"slot={stack_slot_index + 1}/{STACK_SLOT_COUNT}, root={root}, completed={sorted(completed_box_roots)}"
        )
        stack_slot_index = min(stack_slot_index + 1, int(STACK_SLOT_COUNT))

        if _diag_stop_after_release_count_reached_164(release_count):
            print(
                f"[DIAG_STOP_AFTER_RELEASE_164] release_count={release_count} reached. "
                "Stopping after configured release-count diagnosis. APalt lower/home return skipped."
            )
            return release_count

        # 3개 이상 테스트에서는 1층 2개 완료 후 APalt를 상자 높이만큼 내린다.
        if (
            int(FORKLIFT_TRIGGER_COUNT) >= 3
            and release_count == 2
            and int(stack_platform_lower_count) < 1
            and bool(STACK_LOWER_SUPPORT_AFTER_EACH_LAYER)
        ):
            wait_after_release_before_apalt_lower_160(stage_now, label=f"release_count={release_count}")
            print("[APALT_LOWER_REQUEST_145] 3개 이상 적재 모드: 1층 2개 완료 → APalt를 상자 높이만큼 내립니다.")
            ok_lower = slow_lower_apalt_after_first_layer_145(
                stage_now,
                box_height=float(globals().get("BOX_STACK_HEIGHT_145", 0.24)),
            )
            if ok_lower:
                stack_platform_lower_count += 1
            else:
                print("[APALT_LOWER_FAIL_145] APalt lowering 실패. 로그 확인 필요")

        zone_counts_by_root.pop(task.box_move_path, None)
        check_and_trigger_forklift()
        return release_count

    def reset_attempt_state(reason="next attempt", ignore_after_success=False):
        """
        world.pause() 없이 다음 시도 대기 상태로 되돌린다.
        박스 위치는 코드로 되돌리지 않고, 다음 stable bbox를 다시 읽어서 pick target을 갱신한다.
        """
        nonlocal attached, ever_attached, released, returning_home
        nonlocal home_start_joints, home_target_joints, home_return_step
        nonlocal retry_logged, best_attach_reason, best_attach_dist
        nonlocal attached_center_offset, attach_center_z, attach_steps
        nonlocal custom_carry_active, custom_carry_phase, custom_phase_step, custom_targets
        nonlocal custom_fixed_orientation, custom_min_center_z, custom_phase_index
        nonlocal box_stopped_for_pick, pick_started, task_done
        nonlocal pre_attach_align_active, pre_attach_align_phase, pre_attach_align_step, pre_attach_fixed_orientation
        nonlocal loop_cycle_index, loop_attempt_index, ignore_released_center, ignore_wait_counter
        nonlocal stack_platform_lower_count

        stage_now = omni.usd.get_context().get_stage()
        log_oribox_pose_tracker(stage_now, f"reset_attempt_before_restore:{reason}", force=True)

        # 다음 사이클 시작 전 물리 joint/carry 상태 복구
        # 66_: 완료된 박스는 절대 다시 dynamic/physics restore하지 않는다.
        #      이 구간에서 child rigid body가 parent root와 분리되며 순간이동처럼 보이는 현상을 막는다.
        try:
            release_physics_attach_joint(stage_now, reason="reset_attempt_state")
            completed_id = get_task_completed_root_path(task)
            if completed_id in completed_box_roots or str(task.box_path) in completed_box_roots:
                print(f"[LOOP] completed box locked. physics restore skipped: root={completed_id}, box={task.box_path}")
                zero_subtree_velocity(stage_now, task.box_path)
            else:
                set_box_scripted_carry_mode(stage_now, task.box_move_path, False, reenable_physics=False, verbose=False)
                set_prim_kinematic(stage_now, task.stop_check_path, True)
                zero_subtree_velocity(stage_now, task.box_move_path)
        except Exception as exc:
            print(f"[LOOP_WARN] carry mode restore failed: {exc}")

        if ignore_after_success and LOOP_IGNORE_RELEASED_BOX_UNTIL_MOVED:
            bbox_now = get_world_bbox_info(stage_now, task.box_path)
            if bbox_now is not None:
                ignore_released_center = np.array(bbox_now["center"], dtype=float)
                ignore_wait_counter = 0
                print(
                    f"[LOOP] released box ignore anchor=({ignore_released_center[0]:.3f},{ignore_released_center[1]:.3f},{ignore_released_center[2]:.4f}), "
                    f"move_tol={LOOP_RELEASED_BOX_IGNORE_MOVE_TOL:.3f}m"
                )
        else:
            ignore_released_center = None
            ignore_wait_counter = 0

        controller.reset()
        try:
            cart_controller.reset()
        except Exception:
            pass
        box_stop_detector.reset()
        pick_zone_detector.reset()
        try:
            zone_counts_by_root.clear()
        except Exception:
            pass

        attached = False
        ever_attached = False
        released = False
        returning_home = False
        home_start_joints = None
        home_target_joints = None
        home_return_step = 0
        retry_logged = False
        best_attach_reason = "none"
        best_attach_dist = 999.0
        attached_center_offset = None
        attach_center_z = None
        attach_steps = 0
        custom_carry_active = False
        custom_carry_phase = None
        custom_phase_step = 0
        custom_targets = None
        custom_fixed_orientation = None
        custom_min_center_z = None
        custom_phase_index = 0
        box_stopped_for_pick = False
        pick_started = False
        pre_attach_align_active = False
        pre_attach_align_phase = "idle"
        pre_attach_align_step = 0
        pre_attach_fixed_orientation = None
        task_done = False
        loop_cycle_index += 1
        loop_attempt_index += 1

        print(
            f"[LOOP] reset attempt state. cycle={loop_cycle_index}, attempt={loop_attempt_index}, reason={reason}\n"
            "       robot is home/standing. waiting for stopped Small_Cardboard_box again."
        )
        log_oribox_pose_tracker(stage_now, f"reset_attempt_after_restore:{reason}", force=True)

    def start_home_return(reason, success=False):
        """성공/실패 모두 pause하지 않고 home 복귀 후 다음 cycle로 넘긴다."""
        nonlocal returning_home, home_start_joints, home_target_joints, home_return_step
        nonlocal custom_carry_active, custom_carry_phase, custom_phase_step, custom_targets
        nonlocal home_return_reason, home_return_is_success
        nonlocal attached, attached_center_offset, attach_center_z, attach_steps
        nonlocal task_done

        # 134_: release 후 home-return 루프가 실행되도록 task_done을 반드시 풀어준다.
        # 기존 130에서는 release 직후 task_done=True가 먼저 걸리면
        # 메인 루프의 `if is_playing and not task_done:` 조건에 막혀 초기 자세 복귀가 실행되지 않을 수 있었다.
        task_done = False

        home_return_reason = str(reason)
        home_return_is_success = bool(success)
        custom_carry_active = False
        custom_carry_phase = None
        custom_phase_step = 0
        custom_targets = None

        # 실패 루트에서 혹시 attach가 일부 켜져 있었다면 안전하게 release 상태로 돌린다.
        if not success and attached:
            stage_now = omni.usd.get_context().get_stage()
            release_physics_attach_joint(stage_now, reason="failed_home_return")
            set_box_scripted_carry_mode(stage_now, task.box_move_path, False, reenable_physics=True, verbose=True)
            zero_subtree_velocity(stage, task.box_move_path)
            attached = False
            attached_center_offset = None
            attach_center_z = None
            attach_steps = 0

        returning_home = True
        home_start_joints = np.array(robot.get_joint_positions(), dtype=float)
        home_target_joints = np.zeros_like(home_start_joints) if HOME_TARGET_JOINTS_CONFIG is None else np.array(HOME_TARGET_JOINTS_CONFIG, dtype=float)
        home_return_step = 0
        stage_now = omni.usd.get_context().get_stage()
        log_oribox_pose_tracker(stage_now, f"start_home_return:{reason}", force=True)
        print(f"[LOOP_RETURN_134] reason={home_return_reason}, success={home_return_is_success}. returning robot home/initial pose. task_done={task_done}")

    def begin_grid_physics_attach(close_reason, suction_pos):
        """53_: pre-align 또는 기존 controller 경로에서 공통으로 쓰는 물리 흡착 시작 함수."""
        nonlocal attached, ever_attached, attached_center_offset, attach_center_z, attach_steps
        nonlocal custom_carry_active, custom_carry_phase, custom_phase_step, custom_targets
        nonlocal custom_fixed_orientation, custom_min_center_z, custom_phase_index
        nonlocal task_done

        stage_now = omni.usd.get_context().get_stage()
        bbox_for_attach = get_world_bbox_info(stage_now, task.box_path)
        if bbox_for_attach is None:
            print("[PHYSICS_ATTACH_ABORT] bbox_for_attach를 읽지 못했습니다.")
            if RUN_CONTINUOUS_LOOP and LOOP_RETRY_AFTER_ATTACH_FAIL:
                start_home_return("bbox_missing_before_attach", success=False)
            else:
                task_done = True
                my_world.pause()
            return False

        attach_center = np.array(bbox_for_attach["center"], dtype=float)
        grid_attach_center = _LAST_SUCTION_GRID_INFO.get("attach_center")
        if grid_attach_center is None:
            grid_attach_center = np.array(bbox_for_attach["top_center"], dtype=float)
            grid_attach_center[2] = float(grid_attach_center[2] + PHYSICS_ATTACH_TOP_SURFACE_EPS)
        grid_attach_center = np.array(grid_attach_center, dtype=float)
        top_center_for_attach = np.array(bbox_for_attach["top_center"], dtype=float)
        top_gap_at_attach = float(np.array(suction_pos, dtype=float)[2] - top_center_for_attach[2])
        if top_gap_at_attach > float(BOX_ATTACH_Z_MAX) + 1e-6:
            print(
                f"[PHYSICS_ATTACH_ABORT] suction is not on real top surface. "
                f"z_gap={top_gap_at_attach:.4f} > {BOX_ATTACH_Z_MAX:.4f}. retrying, no joint created."
            )
            if RUN_CONTINUOUS_LOOP and LOOP_RETRY_AFTER_ATTACH_FAIL:
                start_home_return("top_surface_gap_too_large_before_attach", success=False)
            else:
                task_done = True
                my_world.pause()
            return False
        # FixedJoint는 실제 윗면 좌표에 생성하지만, carry 계산용 offset은 현재 suction 중심 기준으로 유지한다.
        attached_center_offset = attach_center - np.array(suction_pos, dtype=float)

        joint_ok = True
        if PHYSICS_FIXED_JOINT_ATTACH_ENABLED:
            joint_ok = create_physics_attach_joint(stage_now, task.box_path, grid_attach_center)
        if not joint_ok:
            print("[PHYSICS_ATTACH_ABORT] FixedJoint 생성 실패. 박스 물리를 끄는 fallback은 사용하지 않고 재시도합니다.")
            if RUN_CONTINUOUS_LOOP and LOOP_RETRY_AFTER_ATTACH_FAIL:
                start_home_return("physics_joint_attach_failed", success=False)
            else:
                task_done = True
                my_world.pause()
            return False

        if not PHYSICS_FIXED_JOINT_ATTACH_ENABLED:
            # 56_: 실제 윗면 흡착 판정 후 FixedJoint 대신 top-lock 방식으로 운반한다.
            # collision/rigidBody를 끄지 않고 kinematic만 켜서 박스가 구르거나 튀지 않게 한다.
            set_prim_kinematic(stage, task.box_path, True)
            print("  [CENTER_TOP_LOCK_ATTACH] FixedJoint OFF fallback path. 물리 테스트에서는 이 로그가 나오면 안 됩니다.")

        zero_subtree_velocity(stage, task.box_move_path)
        attached = True
        ever_attached = True
        attach_center_z = float(attach_center[2])
        attach_steps = 0
        print(f"-surface on- OriBoxA_ box attach reason={close_reason}")
        print(f"             grid_attach_center={grid_attach_center}")
        print(f"             attached_center_offset={attached_center_offset}")
        try:
            _slot_idx_attach_164 = int(getattr(task, "_stack_slot_index", stack_slot_index))
            axis_yaw_diagnostic_164(stage_now, task.box_move_path, _slot_idx_attach_164, label="after_attach_fixed_joint")
        except Exception as _axis_attach_exc_164:
            print(f"[BOX_AXIS_YAW_164][WARN] after_attach failed: {type(_axis_attach_exc_164).__name__}: {_axis_attach_exc_164}")

        if CUSTOM_CARRY_AFTER_ATTACH:
            custom_carry_active = True
            custom_phase_step = 0
            custom_phase_index = 0
            custom_min_center_z = float(attach_center_z)

            if CUSTOM_CARRY_MODE == "JOINT_SWING":
                custom_fixed_orientation = None
                custom_targets = make_joint_swing_carry_targets(
                    stage_now,
                    robot,
                    task,
                    attach_center=attach_center,
                    attach_suction_pos=suction_pos,
                    attached_center_offset=attached_center_offset,
                )
                custom_carry_phase = custom_targets["phase_sequence"][0].get("name", "joint_lift")
                print(
                    "[JOINT_CARRY_60_CENTER_BOXAPROP_PLACE] start. "
                    f"center_path={custom_targets.get('center_path')}, "
                    f"lift_delta={custom_targets['lift_delta']}, "
                    f"swing_delta={custom_targets['swing_delta']}, "
                    f"mirror_xy_estimate={custom_targets['mirror_xy_estimate']}, "
                    f"phases={[p['name'] for p in custom_targets.get('phase_sequence', [])]}"
                )
            else:
                custom_carry_phase = "lift"
                custom_fixed_orientation = get_current_ee_pose(robot)[1]
                custom_targets = make_custom_carry_targets(
                    stage_now,
                    task,
                    attach_center=attach_center,
                    attach_suction_pos=suction_pos,
                    attached_center_offset=attached_center_offset,
                )
                if custom_targets.get("phase_sequence"):
                    custom_carry_phase = custom_targets["phase_sequence"][0].get("name", "lift")
                cart_controller.reset()
                print(
                    "[SIMPLE_3STEP_117] start. step1 vertical lift -> step2 joint_1/link_1 rotate -> step3 reverse lower. "
                    f"safe_z={custom_targets['safe_z']:.3f}, "
                    f"lift_suction={custom_targets['lift_suction']}, "
                    f"move_suction={custom_targets['move_suction']}, "
                    f"lower_suction={custom_targets['lower_suction']}, "
                    f"place_center={custom_targets['place_center']}, "
                    f"vertical_only={custom_targets.get('vertical_only', False)}, "
                    f"phases={[p['name'] for p in custom_targets.get('phase_sequence', [])]}"
                )
        return True

    while simulation_app.is_running():
        my_world.step(render=True)
        time.sleep(0.01)
        is_playing = my_world.is_playing()

        if is_playing:
            stage_for_pose_track = omni.usd.get_context().get_stage()
            log_oribox_pose_tracker(stage_for_pose_track, "main_loop", force=False)
            if bool(globals().get("FORKLIFT_TRACK_MAIN_LOOP_139", False)):
                log_forklift_pose_135("main_loop", force=False)

            # 127_ watchdog release:
            # 126에서는 DROP_RELEASE_EXTRA_Z_125=0.10으로 release 높이는 올라갔지만,
            # phase 완료 블록에 못 들어가면 FixedJoint 제거 코드가 실행되지 않았다.
            # 이 블록은 상자가 이미 pick 위치에서 충분히 이동했고 높은 위치에 머무르면
            # custom phase 상태와 무관하게 흡착 joint를 제거한다.
            if (
                bool(globals().get("FORCE_DROP_RELEASE_WATCHDOG_127", True))
                and bool(attached)
                and (not bool(released))
                and str(CUSTOM_CARRY_MODE) == "VERTICAL_JOINT1_REVERSE"
            ):
                try:
                    bbox_watch_127 = get_world_bbox_info(stage_for_pose_track, task.box_path)
                    c_watch_127 = np.array(bbox_watch_127["center"], dtype=float) if bbox_watch_127 is not None else None
                    pick_ref_127 = np.array(task.pick_center, dtype=float) if task.pick_center is not None else None
                    if c_watch_127 is not None and pick_ref_127 is not None:
                        moved_xy_127 = float(np.linalg.norm(c_watch_127[:2] - pick_ref_127[:2]))
                        z_above_pick_127 = float(c_watch_127[2] - pick_ref_127[2])
                        if (
                            moved_xy_127 >= float(FORCE_DROP_RELEASE_MIN_XY_MOVE_127)
                            and z_above_pick_127 >= float(FORCE_DROP_RELEASE_MIN_Z_ABOVE_PICK_127)
                        ):
                            release_watchdog_counter_127 += 1
                        else:
                            release_watchdog_counter_127 = 0

                        if release_watchdog_counter_127 >= int(FORCE_DROP_RELEASE_WATCHDOG_STEPS_127):
                            print(
                                f"[WATCHDOG_DROP_RELEASE_127] trigger: moved_xy={moved_xy_127:.3f}m, "
                                f"z_above_pick={z_above_pick_127:.3f}m, counter={release_watchdog_counter_127}"
                            )
                            release_ok_127 = release_physics_attach_joint(stage_for_pose_track, reason="watchdog_drop_release_127")
                            attached = False
                            released = True
                            custom_carry_active = False
                            custom_carry_phase = None
                            attached_center_offset = None
                            attach_center_z = None
                            task_done = True
                            if MULTI_ORIBOX_STACKING_ENABLED:
                                handle_stack_release_count_145(stage_for_pose_track, task, label="watchdog_drop_release_127")
                            else:
                                completed_box_roots.add(get_task_completed_root_path(task))
                                check_and_trigger_forklift()

                            print(
                                f"-surface off- WATCHDOG_DROP_RELEASE_127: FixedJoint removed={release_ok_127}. "
                                "APalt transform was NOT changed. Box should fall if it is dynamic and not supported."
                            )
                            log_oribox_pose_tracker(stage_for_pose_track, "watchdog_after_drop_release_127_begin", force=True)

                            observe_steps_127 = int(max(0, FORCE_DROP_RELEASE_OBSERVE_STEPS_127))
                            log_interval_127 = int(max(1, FORCE_DROP_RELEASE_LOG_INTERVAL_127))
                            for drop_i_127 in range(observe_steps_127):
                                try:
                                    update_vgc10_suction_anchor(robot)
                                except Exception:
                                    pass
                                my_world.step(render=True)
                                if drop_i_127 % log_interval_127 == 0 or drop_i_127 == observe_steps_127 - 1:
                                    log_oribox_pose_tracker(stage_for_pose_track, f"watchdog_drop_release_127:{drop_i_127}", force=True)

                            if bool(RETURN_HOME_AFTER_WATCHDOG_DROP_127):
                                start_home_return("watchdog_drop_release_127_after_observe", success=True)
                            was_playing = is_playing
                            continue
                except Exception as e:
                    print(f"[WATCHDOG_DROP_RELEASE_127_WARN] {type(e).__name__}: {e}")

        if is_playing and not was_playing:
            my_world.reset()
            stage = omni.usd.get_context().get_stage()
            remove_old_gripper(stage, verbose=False)
            initialize_robot_for_conveyor(robot, my_world)

            # 18_: 이전 run에서 scripted carry 때문에 남은 kinematic/collision off 상태를 먼저 복구한다.
            set_box_scripted_carry_mode(stage, task.box_move_path, False, reenable_physics=True, verbose=True)
            set_prim_kinematic(stage, task.stop_check_path, False)
            zero_subtree_velocity(stage, task.box_move_path)

            # 20_: 박스 이동 자체는 컨베이어가 담당한다.
            # Stop→Play 때도 박스 위치를 코드로 강제 복구하지 않는다.
            if RESET_BOX_TO_USD_START_ON_PLAY:
                set_prim_world_translation(stage, task.box_move_path, task._box_initial_root_pos)
                zero_subtree_velocity(stage, task.box_move_path)

            controller.reset()
            attached = False
            ever_attached = False
            released = False
            returning_home = False
            home_start_joints = None
            home_target_joints = None
            home_return_step = 0
            retry_logged = False
            best_attach_reason = "none"
            best_attach_dist = 999.0
            attached_center_offset = None
            attach_center_z = None
            attach_steps = 0
            custom_carry_active = False
            custom_carry_phase = None
            custom_phase_step = 0
            custom_targets = None
            custom_fixed_orientation = None
            custom_min_center_z = None
            custom_phase_index = 0
            cart_controller.reset()
            box_stop_detector.reset()
            pick_zone_detector.reset()
            completed_box_roots = set()
            zone_counts_by_root = {}
            stack_slot_index = 0
            stack_platform_lower_count = 0
            stack_complete_logged = False
            box_stopped_for_pick = False
            pick_started = False
            pre_attach_align_active = False
            pre_attach_align_phase = "idle"
            pre_attach_align_step = 0
            pre_attach_fixed_orientation = None
            task_done = False

            print("-surface off- reset")
            print("[76_] Conveyor_lift_test clean + APalt pallet + root-coordinate carry")
            print(f"     pick manual offset = {PICK_TARGET_MANUAL_OFFSET}  # 53_: pre-align 사용, manual offset은 기본 0")
            print(f"     pre-align = enabled={PRE_ATTACH_ALIGN_ENABLED}, xy_tol={PRE_ATTACH_ALIGN_XY_TOL}, high_gap={PRE_ATTACH_ALIGN_HEIGHT}, contact_gap={PRE_ATTACH_CONTACT_GAP}, z_gap_ok=[{BOX_ATTACH_Z_MIN},{BOX_ATTACH_Z_MAX}], keep_orientation={PRE_ATTACH_KEEP_CURRENT_ORIENTATION}")
            print(f"     suction grid eval = world_xy={SUCTION_GRID_EVALUATE_AS_WORLD_XY_GRID}, marker_root={SUCTION_GRID_WORLD_MARKER_ROOT_PATH}")
            print(f"     return home immediately after attach = {RETURN_HOME_IMMEDIATELY_AFTER_ATTACH}")
            print(f"     reset box to USD start on play = {RESET_BOX_TO_USD_START_ON_PLAY}  # False면 박스 위치를 코드로 안 움직임")
            print(f"     trigger mode = {PICK_TRIGGER_MODE} / ready_zone_A_stop={PICK_ZONE_REQUIRE_STOPPED_BEFORE_PICK}")
            print(f"     pick zone    = path={PICK_ZONE_PATH}, center={PICK_ZONE_CENTER}, size={PICK_ZONE_SIZE}, required_steps={PICK_ZONE_REQUIRED_STEPS}, zero_velocity_on_start={PICK_ZONE_ZERO_BOX_VELOCITY_ON_PICK_START}")
            print(f"     stack mode   = enabled={MULTI_ORIBOX_STACKING_ENABLED}, parent={ORIBOX_STACK_PARENT_PATH}, prefixes={ORIBOX_STACK_NAME_PREFIXES}, support=/World/APalt, columns={STACK_COLUMNS}, layers={STACK_LAYERS}, slots={STACK_SLOT_COUNT}, axis={STACK_SLOT_AXIS}, lower_platform={STACK_LOWER_SUPPORT_AFTER_EACH_LAYER}")
            print(f"     pose tracker = enabled={POSE_TRACK_ENABLED}, roots={POSE_TRACK_ROOT_PATHS}, every_step={POSE_TRACK_LOG_EVERY_STEP}, interval={POSE_TRACK_LOG_INTERVAL}, jump_tol={POSE_TRACK_JUMP_WARN_TOL}m")
            print(f"     legacy ready zone: y <= {BOX_READY_MIN_Y}, center_z <= {BOX_READY_MAX_CENTER_Z}")
            print(f"     custom carry mode = {CUSTOM_CARRY_MODE}, safe suction z = {CUSTOM_CARRY_SAFE_SUCTION_Z}m, lift delta={CUSTOM_LIFT_DELTA_Z}m, support mesh=/World/APalt, snap=False, upright_orientation_lock=True, drop_release_124=True")
            print(f"     joint carry 30_loop_28_fast_vertical_first: lift_steps={JOINT_LIFT_STEPS}, extra_hold={JOINT_LIFT_EXTRA_HOLD_STEPS}, lift_sign={JOINT_LIFT_SIGN}, j2_delta={JOINT_LIFT_J2_DELTA_RAD}, j3_delta={JOINT_LIFT_J3_DELTA_RAD if JOINT_USE_J3_FOR_LIFT else 0.0}, swing_sign={JOINT_SWING_SIGN}, swing_delta={JOINT_SWING_DELTA_RAD}, clamp={JOINT_SWING_CLAMP_RAD}")
            print(f"     target box      = {task.box_path}")
            print(f"     move wrapper    = {task.box_move_path}")
            print(f"     stop check only = {task.stop_check_path}")
            print(f"     screenshot local translate 참고 = {BOX_SCREENSHOT_LOCAL_TRANSLATE}")
            print(f"     stop gate mode  = bbox_move:{BOX_STOP_USE_BBOX_MOVE}, linear_vel:{BOX_STOP_USE_LINEAR_VEL}, angular_vel:{BOX_STOP_USE_ANGULAR_VEL}")
            print("     8_ note: ang_vel 값이 커도 기본값에서는 INFO_ONLY라 정지 판정을 막지 않음")
            log_oribox_pose_tracker(stage, "after_play_reset", force=True)

        if is_playing and not task_done:
            stage = omni.usd.get_context().get_stage()

            # ------------------------------------------------------------
            # 8_ 핵심: 선택된 /World/OriBoxA/Small_Cardboard_box prim 하나만 정지 판정 대상으로 사용한다.
            # Conveyor 위에서 박스가 이동 중이면 bbox center 변화량이 계속 크므로 여기서 대기한다.
            # 멈춘 순간의 현재 박스 위치를 pick 위치로 다시 확정한다.
            # ------------------------------------------------------------
            if WAIT_UNTIL_BOX_STOPPED_BEFORE_PICK and (not box_stopped_for_pick) and (not attached) and (not released):
                stack_candidate = None
                stack_candidate_ready = False
                stack_candidate_reason = "multi_stack_disabled"

                if MULTI_ORIBOX_STACKING_ENABLED:
                    if len(completed_box_roots) >= int(FORKLIFT_TRIGGER_COUNT):
                        hold_robot_current_pose(robot)
                        update_vgc10_suction_anchor(robot)
                        if not stack_complete_logged:
                            print(f"[STACK_TRIGGER_COMPLETE_145] release_count={len(completed_box_roots)}/{FORKLIFT_TRIGGER_COUNT}. APalt 운반 조건을 만족했으므로 추가 pick을 막습니다.")
                            stack_complete_logged = True
                        was_playing = is_playing
                        continue

                    if stack_slot_index >= int(STACK_SLOT_COUNT) and not bool(globals().get("ALLOW_CONTINUE_AFTER_STACK_FULL_137", False)):
                        hold_robot_current_pose(robot)
                        update_vgc10_suction_anchor(robot)
                        if not stack_complete_logged:
                            print(f"[STACK_COMPLETE] {STACK_COLUMNS}x{STACK_LAYERS} = {STACK_SLOT_COUNT}개 slot 적재 완료. 더 이상 OriBoxA_ 박스를 집지 않습니다.")
                            stack_complete_logged = True
                        was_playing = is_playing
                        continue
                    elif stack_slot_index >= int(STACK_SLOT_COUNT) and bool(globals().get("ALLOW_CONTINUE_AFTER_STACK_FULL_137", False)):
                        if not stack_complete_logged:
                            print(f"[STACK_CONTINUE_137] stack_slot_index={stack_slot_index}, slot_count={STACK_SLOT_COUNT}. forklift/continuous test라서 stack full 정지를 무시합니다.")
                            stack_complete_logged = True

                    stack_candidate, stack_candidate_ready, stack_candidate_reason = select_oriboxa_candidate_in_front_zone(
                        stage,
                        completed_roots=completed_box_roots,
                        zone_counts_by_root=zone_counts_by_root,
                    )
                    if stack_candidate is not None:
                        task.set_active_box(
                            stack_candidate["root_path"],
                            stack_candidate["box_path"],
                            stack_slot_index=stack_slot_index,
                        )
                        bbox_wait = stack_candidate["bbox"]
                    else:
                        bbox_wait = None
                else:
                    bbox_wait = get_world_bbox_info(stage, task.stop_check_path)

                suction_wait = update_vgc10_suction_anchor(robot)
                hold_robot_current_pose(robot)

                # 30_: 성공 후 방금 내려놓은 같은 박스를 바로 다시 집지 않도록,
                # release anchor에서 충분히 멀어질 때까지 다음 pick을 막는다.
                if LOOP_IGNORE_RELEASED_BOX_UNTIL_MOVED and ignore_released_center is not None and bbox_wait is not None:
                    cur_center = np.array(bbox_wait["center"], dtype=float)
                    moved_xy = float(np.linalg.norm(cur_center[:2] - np.array(ignore_released_center, dtype=float)[:2]))
                    if moved_xy < float(LOOP_RELEASED_BOX_IGNORE_MOVE_TOL):
                        ignore_wait_counter += 1
                        box_stop_detector.reset()
                        pick_zone_detector.reset()
                        if ignore_wait_counter % LOOP_WAIT_LOG_INTERVAL == 0:
                            print(
                                f"  [loop_ignore_released_box] moved_xy={moved_xy:.4f}/{LOOP_RELEASED_BOX_IGNORE_MOVE_TOL:.4f}. "
                                "waiting until this prim moves away or a new box arrives."
                            )
                        was_playing = is_playing
                        continue
                    else:
                        print(
                            f"[LOOP] released-box ignore cleared. moved_xy={moved_xy:.4f} >= {LOOP_RELEASED_BOX_IGNORE_MOVE_TOL:.4f}."
                        )
                        ignore_released_center = None
                        ignore_wait_counter = 0
                        box_stop_detector.reset()
                        pick_zone_detector.reset()

                if PICK_TRIGGER_MODE == "FRONT_ZONE":
                    if MULTI_ORIBOX_STACKING_ENABLED:
                        zone_ready, zone_reason = stack_candidate_ready, stack_candidate_reason
                    else:
                        zone_ready, zone_reason = pick_zone_detector.update(bbox_wait)

                    stopped_now = False
                    stop_reason = "stop_check_waiting_for_zone"
                    if zone_ready and bool(PICK_ZONE_REQUIRE_STOPPED_BEFORE_PICK):
                        stopped_now, stop_reason = box_stop_detector.update(stage, task.stop_check_path, bbox_wait)
                    elif zone_ready:
                        stopped_now = True
                        stop_reason = "stop_check_disabled"
                    else:
                        box_stop_detector.reset()

                    log_front_zone = PICK_ZONE_LOG_EVERY_STEP or (pick_zone_detector.total_steps % PICK_ZONE_LOG_INTERVAL == 0) or zone_ready
                    if log_front_zone:
                        if bbox_wait is not None:
                            c = bbox_wait["center"]
                            t = bbox_wait["top_center"]
                            print(
                                f"  [wait_ready_zone_A] {zone_reason} | {stop_reason} "
                                f"box_center=({c[0]:.3f},{c[1]:.3f},{c[2]:.4f}) "
                                f"box_top=({t[0]:.3f},{t[1]:.3f},{t[2]:.4f}) "
                                f"suction=({suction_wait[0]:.3f},{suction_wait[1]:.3f},{suction_wait[2]:.4f})"
                            )
                        else:
                            print(f"  [wait_ready_zone_A] {zone_reason} | {stop_reason}")

                    if zone_ready and stopped_now:
                        if not set_task_pick_from_current_box(task, stage, bbox_wait):
                            print("[FAIL] ready zone A + stopped passed but pick position update failed. pausing world.")
                            task_done = True
                            my_world.pause()
                            was_playing = is_playing
                            continue

                        if PICK_ZONE_ZERO_BOX_VELOCITY_ON_PICK_START:
                            zero_subtree_velocity(stage, task.box_move_path)

                        controller.reset()
                        try:
                            cart_controller.reset()
                        except Exception:
                            pass
                        box_stopped_for_pick = True  # 이후 should_attach_oribox의 gate를 통과시키기 위한 내부 flag
                        pick_started = True
                        pre_attach_align_active = bool(PRE_ATTACH_ALIGN_ENABLED)
                        pre_attach_align_phase = "xy"
                        pre_attach_align_step = 0
                        pre_attach_fixed_orientation = get_current_ee_pose(robot)[1] if bool(PRE_ATTACH_KEEP_CURRENT_ORIENTATION) else None
                        try:
                            yaw_diagnostic_against_slot_163(stage, task.box_move_path, int(stack_slot_index), label="pre_pick_selected")
                            axis_yaw_diagnostic_164(stage, task.box_move_path, int(stack_slot_index), label="pre_pick_selected")
                        except Exception as _pre_yaw_exc_163:
                            print(f"[PRE_PICK_YAW_163][WARN] failed: {type(_pre_yaw_exc_163).__name__}: {_pre_yaw_exc_163}")
                        print(
                            f"-ready zone A OriBoxA_ stopped target detected- pre-align enabled root={get_task_completed_root_path(task)} slot={stack_slot_index + 1}/{STACK_SLOT_COUNT} orientation_mode={'fixed_current' if pre_attach_fixed_orientation is not None else 'free_position_only'} "
                            f"box_center=({task.box_initial_center[0]:.3f},{task.box_initial_center[1]:.3f},{task.box_initial_center[2]:.4f}) "
                            f"box_top=({task.box_initial_top_center[0]:.3f},{task.box_initial_top_center[1]:.3f},{task.box_initial_top_center[2]:.4f}) "
                            f"pick_target=({task.pick_center[0]:.3f},{task.pick_center[1]:.3f},{task.pick_center[2]:.4f}) "
                            f"goal_center=({task.goal_center[0]:.3f},{task.goal_center[1]:.3f},{task.goal_center[2]:.4f})"
                        )
                        log_oribox_pose_tracker(stage, f"ready_zone_A_selected:{task.box_move_path}", force=True)

                    was_playing = is_playing
                    continue

                stopped_now, stop_reason = box_stop_detector.update(stage, task.stop_check_path, bbox_wait)

                if BOX_STOP_LOG_EVERY_STEP or (box_stop_detector.total_steps % BOX_STOP_LOG_INTERVAL == 0) or stopped_now:
                    if bbox_wait is not None:
                        c = bbox_wait["center"]
                        t = bbox_wait["top_center"]
                        print(
                            f"  [wait_box_stop] {stop_reason} "
                            f"box_center=({c[0]:.3f},{c[1]:.3f},{c[2]:.4f}) "
                            f"box_top=({t[0]:.3f},{t[1]:.3f},{t[2]:.4f}) "
                            f"suction=({suction_wait[0]:.3f},{suction_wait[1]:.3f},{suction_wait[2]:.4f})"
                        )
                    else:
                        print(f"  [wait_box_stop] {stop_reason}")

                if stopped_now:
                    ready_ok, ready_reason = is_box_ready_for_pick_zone(bbox_wait)
                    if not ready_ok:
                        if bbox_wait is not None:
                            c = bbox_wait["center"]
                            print(
                                f"  [wait_box_ready] stable_but_not_pick_zone reason={ready_reason} "
                                f"box_center=({c[0]:.3f},{c[1]:.3f},{c[2]:.4f}). 계속 컨베이어 이동 대기."
                            )
                        box_stop_detector.reset()
                        pick_zone_detector.reset()
                        was_playing = is_playing
                        continue

                    if not set_task_pick_from_current_box(task, stage, bbox_wait):
                        print("[FAIL] box stopped check passed but pick position update failed. pausing world.")
                        task_done = True
                        my_world.pause()
                        was_playing = is_playing
                        continue

                    # 18_: 박스는 컨베이어가 멈춰준 상태를 그대로 사용한다. 강제 freeze는 하지 않는다.
                    zero_subtree_velocity(stage, task.box_move_path)
                    if BOX_FREEZE_AFTER_STOP:
                        set_prim_kinematic(stage, task.stop_check_path, True)
                        zero_subtree_velocity(stage, task.box_move_path)

                    controller.reset()
                    box_stopped_for_pick = True
                    pick_started = True
                    try:
                        yaw_diagnostic_against_slot_163(stage, task.box_move_path, int(stack_slot_index), label="pre_pick_selected_single")
                        axis_yaw_diagnostic_164(stage, task.box_move_path, int(stack_slot_index), label="pre_pick_selected_single")
                    except Exception as _pre_yaw_exc_163:
                        print(f"[PRE_PICK_YAW_163][WARN] failed: {type(_pre_yaw_exc_163).__name__}: {_pre_yaw_exc_163}")
                    print(
                        f"-selected Small_Cardboard_box stopped- pick enabled "
                        f"box_center=({task.box_initial_center[0]:.3f},{task.box_initial_center[1]:.3f},{task.box_initial_center[2]:.4f}) "
                        f"box_top=({task.box_initial_top_center[0]:.3f},{task.box_initial_top_center[1]:.3f},{task.box_initial_top_center[2]:.4f}) "
                        f"pick_target=({task.pick_center[0]:.3f},{task.pick_center[1]:.3f},{task.pick_center[2]:.4f}) "
                        f"goal_center=({task.goal_center[0]:.3f},{task.goal_center[1]:.3f},{task.goal_center[2]:.4f})"
                    )

                was_playing = is_playing
                continue

            if returning_home:
                # 16_ 핵심: 흡착된 상태로 로봇을 0 joint 초기 자세로 복귀한다.
                # 박스는 VGC10 suction point를 계속 따라가게 유지한다.
                if home_start_joints is None:
                    home_start_joints = np.array(robot.get_joint_positions(), dtype=float)
                if home_target_joints is None:
                    if HOME_TARGET_JOINTS_CONFIG is None:
                        home_target_joints = np.zeros_like(home_start_joints)
                    else:
                        home_target_joints = np.array(HOME_TARGET_JOINTS_CONFIG, dtype=float)

                alpha = min(1.0, float(home_return_step) / float(max(1, HOME_RETURN_STEPS)))
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                target_joints = (1.0 - smooth) * home_start_joints + smooth * home_target_joints
                robot.apply_action(ArticulationAction(joint_positions=target_joints))

                suction_pos_now = update_vgc10_suction_anchor(robot)

                if attached and KEEP_BOX_ATTACHED_DURING_HOME_RETURN:
                    if attached_center_offset is None:
                        bbox_for_follow = get_world_bbox_info(stage, task.box_path)
                        follow_center = np.array(bbox_for_follow["center"], dtype=float) if bbox_for_follow is not None else np.zeros(3)
                        attached_center_offset = follow_center - np.array(suction_pos_now, dtype=float)

                    desired_box_center = np.array(suction_pos_now, dtype=float) + np.array(attached_center_offset, dtype=float)
                    if BOX_DISABLE_PHYSICS_DURING_CARRY:
                        set_box_scripted_carry_mode(stage, task.box_move_path, True, verbose=False)
                    move_box_center_to(
                        stage,
                        task.box_move_path,
                        desired_center=desired_box_center,
                        root_to_center_offset=task.box_root_to_center_offset,
                    )
                    zero_subtree_velocity(stage, task.box_move_path)
                else:
                    zero_prim_velocity(stage, task.box_path)

                bbox_now = get_world_bbox_info(stage, task.box_path)

                if home_return_step % 10 == 0 or alpha >= 1.0:
                    box_center_now = bbox_now["center"] if bbox_now is not None else np.zeros(3)
                    print(
                        f"  [return_home_attach] step={home_return_step}/{HOME_RETURN_STEPS} "
                        f"alpha={alpha:.2f} "
                        f"box=({box_center_now[0]:.3f},{box_center_now[1]:.3f},{box_center_now[2]:.4f}) "
                        f"suction=({suction_pos_now[0]:.3f},{suction_pos_now[1]:.3f},{suction_pos_now[2]:.4f}) "
                        f"surface={'ON' if attached else 'OFF'}"
                    )
                    log_oribox_pose_tracker(stage, f"return_home:{home_return_step}", force=True)

                home_return_step += 1
                if alpha >= 1.0:
                    # 18_/30_: 어떤 경우에도 다음 실행 때 박스가 공중에 붙어 있지 않도록 carry mode를 해제한다.
                    if attached:
                        set_box_scripted_carry_mode(stage, task.box_move_path, False, reenable_physics=True, verbose=True)
                        zero_subtree_velocity(stage, task.box_move_path)
                        attached = False

                    if RUN_CONTINUOUS_LOOP:
                        print(f"[LOOP_HOME_DONE] robot returned home. reason={home_return_reason}, success={home_return_is_success}")
                        reset_attempt_state(
                            reason=f"home_done:{home_return_reason}",
                            ignore_after_success=bool(home_return_is_success),
                        )
                    else:
                        print("[완료] robot returned home. box is not kept attached in the air. pausing world.")
                        task_done = True
                        my_world.pause()

                was_playing = is_playing
                continue

            # ------------------------------------------------------------
            # 53_ 핵심: 작업영역 진입 후 box_top 좌표로 suction 중심을 먼저 맞춘다.
            # 기존 52_ 로그에서 box_top은 읽었지만 suction_x가 약 0.27m 벗어나 grid_hits=0/9였다.
            # 여기서는 PickPlaceController event로 내려가기 전에 RMPFlow로 직접
            #   1) box_top 위 높은 위치에서 XY 정렬
            #   2) 같은 XY에서 Z 접근
            #   3) 9점 흡착 판정 OK면 FixedJoint 생성
            # 순서로 처리한다.
            # ------------------------------------------------------------
            if pre_attach_align_active and (not attached) and (not released):
                bbox_now = get_world_bbox_info(stage, task.box_path)
                if bbox_now is None:
                    print("[PRE_ALIGN_FAIL] bbox를 읽지 못했습니다. home return 후 재시도합니다.")
                    if RUN_CONTINUOUS_LOOP and LOOP_RETRY_AFTER_ATTACH_FAIL:
                        start_home_return("prealign_bbox_missing", success=False)
                    else:
                        task_done = True
                        my_world.pause()
                    was_playing = is_playing
                    continue

                suction_now = update_vgc10_suction_anchor(robot)
                top = np.array(bbox_now["top_center"], dtype=float)
                suction_now = np.array(suction_now, dtype=float)
                xy_err = float(np.linalg.norm(suction_now[:2] - top[:2]))
                z_gap = float(suction_now[2] - top[2])

                if pre_attach_align_phase == "xy":
                    desired_suction = np.array([top[0], top[1], top[2] + float(PRE_ATTACH_ALIGN_HEIGHT)], dtype=float)
                    desired_z_err = float(abs(suction_now[2] - desired_suction[2]))
                    if xy_err <= float(PRE_ATTACH_ALIGN_XY_TOL) and desired_z_err <= float(PRE_ATTACH_ALIGN_Z_TOL):
                        pre_attach_align_phase = "z"
                        try:
                            cart_controller.reset()
                        except Exception:
                            pass
                        print(
                            f"[PRE_ALIGN_PHASE] XY aligned. switching to Z approach. "
                            f"xy_err={xy_err:.4f}, z_gap={z_gap:.4f}, box_top=({top[0]:.3f},{top[1]:.3f},{top[2]:.4f})"
                        )
                else:
                    desired_suction = np.array([top[0], top[1], top[2] + float(PRE_ATTACH_CONTACT_GAP)], dtype=float)

                apply_cartesian_suction_target(
                    cart_controller,
                    robot,
                    current_suction_pos=suction_now,
                    desired_suction_pos=desired_suction,
                    fixed_orientation=pre_attach_fixed_orientation,
                )
                suction_after = update_vgc10_suction_anchor(robot)
                suction_after = np.array(suction_after, dtype=float)
                xy_err_after = float(np.linalg.norm(suction_after[:2] - top[:2]))
                z_gap_after = float(suction_after[2] - top[2])

                grid_ok, grid_summary, _grid_info = evaluate_suction_grid_on_box_top(
                    stage,
                    bbox_now,
                    event="prealign",
                    verbose=bool(PRE_ATTACH_EVALUATE_GRID_EVERY_STEP),
                )

                if (pre_attach_align_step % int(max(1, PRE_ATTACH_LOG_INTERVAL)) == 0) or grid_ok:
                    print(
                        f"  [PRE_ALIGN] step={pre_attach_align_step}/{PRE_ATTACH_MAX_STEPS}, phase={pre_attach_align_phase}, "
                        f"box_center=({bbox_now['center'][0]:.3f},{bbox_now['center'][1]:.3f},{bbox_now['center'][2]:.4f}), "
                        f"box_top=({top[0]:.3f},{top[1]:.3f},{top[2]:.4f}), "
                        f"suction=({suction_after[0]:.3f},{suction_after[1]:.3f},{suction_after[2]:.4f}), "
                        f"target=({desired_suction[0]:.3f},{desired_suction[1]:.3f},{desired_suction[2]:.4f}), "
                        f"xy_err={xy_err_after:.4f}/{PRE_ATTACH_ALIGN_XY_TOL:.4f}, "
                        f"z_gap={z_gap_after:.4f}, grid_ok={grid_ok}"
                    )

                if grid_ok and pre_attach_align_step >= int(PRE_ATTACH_MIN_STEPS_BEFORE_ATTACH):
                    pre_attach_align_active = False
                    pre_attach_align_phase = "done"
                    begin_grid_physics_attach("prealign_" + str(grid_summary), suction_after)
                    was_playing = is_playing
                    continue

                pre_attach_align_step += 1
                if pre_attach_align_step >= int(PRE_ATTACH_MAX_STEPS):
                    print(
                        f"[PRE_ALIGN_FAIL] max steps reached. xy_err={xy_err_after:.4f}, z_gap={z_gap_after:.4f}, "
                        f"grid={grid_summary}. home return 후 재시도합니다."
                    )
                    if RUN_CONTINUOUS_LOOP and LOOP_RETRY_AFTER_ATTACH_FAIL:
                        start_home_return("prealign_timeout", success=False)
                    else:
                        task_done = True
                        my_world.pause()

                was_playing = is_playing
                continue

            if custom_carry_active and attached and not released:
                # 23_ 핵심: attach 이후 PickPlaceController event를 더 진행하지 않는다.
                # 로봇 suction point를 직접 lift -> move -> lower target으로 보낸다.
                suction_now = update_vgc10_suction_anchor(robot)
                bbox_now = get_world_bbox_info(stage, task.box_path)
                box_center_now = bbox_now["center"] if bbox_now is not None else np.zeros(3)

                if custom_targets is None:
                    custom_targets = make_custom_carry_targets(
                        stage,
                        task,
                        attach_center=box_center_now,
                        attach_suction_pos=suction_now,
                        attached_center_offset=attached_center_offset,
                    )

                # 25_ 관절 기반 운반 모드.
                # 기존 cartesian waypoint 방식은 target_suction으로 로봇을 끌고 가면서
                # 베이스/팔을 관통하거나 phase 전환 때 박스가 튀는 문제가 있었다.
                # 여기서는 joint target만 부드럽게 보간하고, 박스는 suction point를 따라오게 한다.
                if isinstance(custom_targets, dict) and custom_targets.get("mode") == "JOINT_SWING":
                    phase_sequence = custom_targets.get("phase_sequence", [])
                    custom_phase_index = int(max(0, min(custom_phase_index, len(phase_sequence) - 1)))
                    phase_info = phase_sequence[custom_phase_index]
                    custom_carry_phase = phase_info.get("name", str(custom_phase_index))
                    duration = int(max(1, phase_info.get("steps", 1)))

                    if "start_joints" not in phase_info:
                        phase_info["start_joints"] = np.array(robot.get_joint_positions(), dtype=float)
                    start_j = np.array(phase_info["start_joints"], dtype=float)
                    target_j = np.array(phase_info["target_joints"], dtype=float)
                    alpha = min(1.0, float(custom_phase_step) / float(duration))
                    smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                    cmd_j = (1.0 - smooth) * start_j + smooth * target_j
                    link2_ref = float(custom_targets.get("link2_j2_ref_rad", cmd_j[1] if len(cmd_j)>1 else 0.0)) if isinstance(custom_targets, dict) else (cmd_j[1] if len(cmd_j)>1 else 0.0)
                    cmd_j = _apply_link2_orient_z_guard_to_joints(cmd_j, link2_ref, label=f"runtime_{custom_carry_phase}")
                    robot.apply_action(ArticulationAction(joint_positions=cmd_j))

                    suction_after = update_vgc10_suction_anchor(robot)
                    if BOX_DISABLE_PHYSICS_DURING_CARRY:
                        set_box_scripted_carry_mode(stage, task.box_move_path, True, verbose=False)

                    min_center_z = None
                    if custom_carry_phase in ("joint_lower", "joint_settle"):
                        min_center_z = float(JOINT_RELEASE_MIN_BOX_CENTER_Z)

                    if PHYSICS_FIXED_JOINT_ATTACH_ENABLED:
                        # 51_: 실제 물리 joint가 박스를 끌고 가게 둔다. 박스 transform 직접 이동 금지.
                        bbox_after = get_world_bbox_info(stage, task.box_path)
                        c_after = bbox_after["center"] if bbox_after is not None else np.array(attach_center, dtype=float)
                        desired_center = c_after
                        ok_follow = True
                    else:
                        ok_follow, desired_center = follow_box_to_suction(
                            stage,
                            task,
                            suction_pos=suction_after,
                            attached_center_offset=attached_center_offset,
                            min_center_z=min_center_z,
                        )
                        bbox_after = get_world_bbox_info(stage, task.box_path)
                        c_after = bbox_after["center"] if bbox_after is not None else desired_center

                    if custom_phase_step % JOINT_CARRY_LOG_INTERVAL == 0 or alpha >= 1.0:
                        link2_ref = float(custom_targets.get("link2_j2_ref_rad", cmd_j[1] if len(cmd_j)>1 else 0.0)) if isinstance(custom_targets, dict) else (cmd_j[1] if len(cmd_j)>1 else 0.0)
                        link2_z_est = _estimate_link2_orient_z_deg_from_j2(cmd_j[1], link2_ref) if len(cmd_j) > 1 else 0.0
                        print(
                            f"  [{custom_carry_phase}] step={custom_phase_step}/{duration}, "
                            f"alpha={alpha:.2f}, "
                            f"j1={cmd_j[0] if len(cmd_j)>0 else 0.0:.3f}, "
                            f"j2={cmd_j[1] if len(cmd_j)>1 else 0.0:.3f}, "
                            f"j3={cmd_j[2] if len(cmd_j)>2 else 0.0:.3f}, "
                            f"link2_z_est={link2_z_est:+.2f}deg/min={LINK2_ORIENT_Z_MIN_DEG:+.2f}, "
                            f"suction=({suction_after[0]:.3f},{suction_after[1]:.3f},{suction_after[2]:.3f}), "
                            f"box=({c_after[0]:.3f},{c_after[1]:.3f},{c_after[2]:.4f}), "
                            f"follow_ok={ok_follow}, top_lock_path={TOP_LOCK_FOLLOW_DESIRED_PATH}"
                        )

                    if bool(globals().get("PHYSICS_FIXED_JOINT_DIAGNOSTIC_NO_LIFT", False)):
                        physics_diag_log_after_attach_step(stage, custom_carry_phase, custom_phase_step)

                    custom_phase_step += 1
                    phase_done = alpha >= 1.0

                    # 27_: 벽에 붙은 박스를 먼저 충분히 위로 빼낸 뒤에만 joint_1 회전 시작.
                    # alpha가 1.0이 되어도 박스/흡착점 높이가 낮으면 joint_lift target을 유지한다.
                    if custom_carry_phase == "joint_lift" and phase_done and JOINT_STRICT_VERTICAL_LIFT_BEFORE_SWING:
                        box_z_now = float(c_after[2]) if c_after is not None else -999.0
                        suction_z_now = float(suction_after[2])
                        lift_ready = (
                            box_z_now >= float(JOINT_SWING_START_MIN_BOX_CENTER_Z)
                            and suction_z_now >= float(JOINT_SWING_START_MIN_SUCTION_Z)
                        )
                        extra_hold_done = custom_phase_step >= (duration + int(JOINT_LIFT_EXTRA_HOLD_STEPS))
                        if not lift_ready and not extra_hold_done:
                            phase_done = False
                            if custom_phase_step % JOINT_CARRY_LOG_INTERVAL == 0:
                                print(
                                    f"  [vertical_lift_gate] hold joint_1. "
                                    f"box_z={box_z_now:.3f}/{JOINT_SWING_START_MIN_BOX_CENTER_Z:.3f}, "
                                    f"suction_z={suction_z_now:.3f}/{JOINT_SWING_START_MIN_SUCTION_Z:.3f}, "
                                    f"extra={custom_phase_step-duration}/{JOINT_LIFT_EXTRA_HOLD_STEPS}"
                                )
                        elif not lift_ready and extra_hold_done:
                            # 무한 대기를 피하기 위한 안전장치. 이 로그가 뜨면 lift delta를 키우거나 threshold를 낮춰야 한다.
                            print(
                                f"  [vertical_lift_gate_WARN] height threshold not fully reached, but extra hold ended. "
                                f"box_z={box_z_now:.3f}/{JOINT_SWING_START_MIN_BOX_CENTER_Z:.3f}, "
                                f"suction_z={suction_z_now:.3f}/{JOINT_SWING_START_MIN_SUCTION_Z:.3f}; "
                                "proceeding to joint_1 swing."
                            )

                    if custom_carry_phase == "joint_lower" and float(suction_after[2]) <= float(JOINT_RELEASE_MIN_SUCTION_Z):
                        print(
                            f"  [joint_lower_guard] suction_z={suction_after[2]:.3f} <= {JOINT_RELEASE_MIN_SUCTION_Z:.3f}; "
                            "stop lowering and release soon."
                        )
                        phase_done = True

                    if phase_done:
                        if custom_phase_index < len(phase_sequence) - 1:
                            custom_phase_index += 1
                            custom_phase_step = 0
                            phase_sequence[custom_phase_index]["start_joints"] = np.array(robot.get_joint_positions(), dtype=float)
                            next_name = phase_sequence[custom_phase_index].get("name", str(custom_phase_index))
                            print(f"[JOINT_CARRY] phase done -> next={next_name}")
                        else:
                            # 102_: 물리 흡착 lift 성공 확인 모드.
                            # 여기서는 바로 release하지 않고, FixedJoint를 유지한 채 pause한다.
                            # 상자가 물리 joint로 실제로 들려 있는지 먼저 눈으로 확인한다.
                            if bool(globals().get("PHYSICS_LIFT_KEEP_ATTACHED_AND_PAUSE", False)):
                                if bool(globals().get("PHYSICS_FIXED_JOINT_DIAGNOSTIC_NO_LIFT", False)):
                                    print("[PHYSICS_DIAG_DONE_105] localRot + mass/inertia diagnostic hold finished. FixedJoint is still attached; no lift/no release/no box xform follow. Pausing now.")
                                    physics_diag_log_after_attach_step(stage, custom_carry_phase, custom_phase_step, force=True)
                                    log_oribox_pose_tracker(stage, "physics_diag_done_104", force=True)
                                else:
                                    print("[PHYSICS_LIFT_SWING_RELEASE_107] lift finished after stabilize. FixedJoint is still attached; no release, no box xform follow, no kinematic carry. Pausing now.")
                                    log_oribox_pose_tracker(stage, "physics_lift_success_hold_102", force=True)
                                my_world.pause()
                                custom_carry_active = False
                                custom_carry_phase = None
                                was_playing = is_playing
                                continue

                            # 75_: joint_1 회전이 끝나면 현재 위치에서 바로 release한다.
                            # snap_box_to_stack_slot_if_enabled()를 호출하지 않는다. 순간이동 방지.
                            release_physics_attach_joint(stage, reason="joint_carry_release")
                            if BOX_DISABLE_PHYSICS_DURING_CARRY:
                                set_box_scripted_carry_mode(
                                    stage,
                                    task.box_move_path,
                                    False,
                                    reenable_physics=BOX_REENABLE_PHYSICS_AFTER_RELEASE,
                                    verbose=True,
                                )
                            elif not PHYSICS_FIXED_JOINT_ATTACH_ENABLED:
                                set_prim_kinematic(stage, task.box_path, False)
                            zero_subtree_velocity(stage, task.box_move_path)
                            if MULTI_ORIBOX_STACKING_ENABLED:
                                handle_stack_release_count_145(stage, task, label="joint_carry_release")
                            attached = False
                            released = True
                            custom_carry_active = False
                            custom_carry_phase = None
                            attached_center_offset = None
                            attach_center_z = None
                            print("-surface off- physics FixedJoint release after lift test. no box xform follow/no kinematic/no snap.")

                            if RETURN_HOME_AFTER_RELEASE:
                                if RUN_CONTINUOUS_LOOP:
                                    start_home_return("success_release_after_joint_carry", success=True)
                                else:
                                    returning_home = True
                                    home_start_joints = np.array(robot.get_joint_positions(), dtype=float)
                                    home_target_joints = np.zeros_like(home_start_joints)
                                    home_return_step = 0
                                    print("[RETURN] box released. returning robot to initial joint pose...")

                    was_playing = is_playing
                    continue

                phase_sequence = custom_targets.get("phase_sequence") if isinstance(custom_targets, dict) else None
                if phase_sequence:
                    custom_phase_index = int(max(0, min(custom_phase_index, len(phase_sequence) - 1)))
                    phase_info = phase_sequence[custom_phase_index]
                    custom_carry_phase = phase_info.get("name", str(custom_phase_index))
                    phase_kind = phase_info.get("kind", "move")

                    # 117_: Hybrid mode special phase.
                    # RMPFlow로 긴 XY 이동을 하지 않고, 현재 관절 자세에서 joint_1/link_1 하나만 회전한다.
                    if phase_kind == "joint1_rotate":
                        duration = int(max(1, phase_info.get("steps", HYBRID_JOINT1_ROTATE_STEPS)))
                        if "start_joints" not in phase_info:
                            start_j = np.array(robot.get_joint_positions(), dtype=float)
                            target_j = start_j.copy()
                            ref_j1 = float(start_j[0]) if len(start_j) > 0 else 0.0

                            # joint_1 회전량은 현재 들어올린 박스 방향 -> 뒤쪽 큐브/place_center 방향의 yaw 차이로 계산한다.
                            # 156_: 단, 기본 배치 회전은 "현재 joint_1 + N도"가 아니라
                            #       "초기화 상태 joint_1 + N도"를 절대 목표로 사용한다.
                            bbox_now = get_world_bbox_info(stage, task.box_path)
                            box_center_now = np.array(bbox_now["center"], dtype=float) if bbox_now is not None else np.array(suction_now, dtype=float)
                            place_center_now = np.array(custom_targets.get("place_center", box_center_now), dtype=float) if isinstance(custom_targets, dict) else box_center_now.copy()
                            robot_center_now, center_path_now = get_robot_center_for_goal(stage)

                            mode = str(phase_info.get("mode", HYBRID_JOINT1_ROTATE_MODE))
                            sign = float(phase_info.get("sign", HYBRID_JOINT1_ROTATE_SIGN))
                            max_rad = abs(math.radians(float(phase_info.get("max_deg", HYBRID_JOINT1_ROTATE_MAX_DEG))))
                            fallback_rad = math.radians(float(phase_info.get("fallback_deg", HYBRID_JOINT1_ROTATE_FALLBACK_DEG)))
                            raw_delta = fallback_rad
                            delta_reason = "fallback_deg"
                            initial_j1 = None
                            absolute_target_j1 = None

                            if mode == "goal_angle" and robot_center_now is not None:
                                rc = np.array(robot_center_now[:2], dtype=float)
                                start_vec = box_center_now[:2] - rc
                                goal_vec = place_center_now[:2] - rc
                                if np.linalg.norm(start_vec) > 1e-6 and np.linalg.norm(goal_vec) > 1e-6:
                                    start_ang = float(np.arctan2(start_vec[1], start_vec[0]))
                                    goal_ang = float(np.arctan2(goal_vec[1], goal_vec[0]))
                                    raw_delta = float((goal_ang - start_ang + np.pi) % (2.0 * np.pi) - np.pi)
                                    delta_reason = f"goal_angle:{center_path_now}"

                            signed_delta = float(sign) * float(raw_delta)

                            # 157_: 정답지 APalt_slot 좌표를 이용해 joint_1 회전량을 자동 계산한다.
                            # x1/x2 = 로봇 base 기준 slot 방향, x3 = 로봇 base 기준 현재 박스 방향.
                            # 현재 자세에서 필요한 회전량 = (slot 방향 - 현재 박스 방향).
                            # 이 회전은 최종 위치 결정용이 아니라 APalt 방향 접근용이며,
                            # 최종 위치는 slot_marker_move_over/lower가 APalt_slot 좌표로 보정한다.
                            auto_slot_used_157 = False
                            if bool(globals().get("AUTO_JOINT1_FROM_SLOT_MARKER_ENABLED_157", False)) and robot_center_now is not None:
                                try:
                                    rc157 = np.array(robot_center_now[:2], dtype=float)
                                    box_vec157 = np.array(box_center_now[:2], dtype=float) - rc157
                                    slot_vec157 = np.array(place_center_now[:2], dtype=float) - rc157
                                    if np.linalg.norm(box_vec157) > 1e-6 and np.linalg.norm(slot_vec157) > 1e-6:
                                        x3_box_angle_157 = float(np.arctan2(box_vec157[1], box_vec157[0]))
                                        x_slot_angle_157 = float(np.arctan2(slot_vec157[1], slot_vec157[0]))
                                        raw_auto_delta_157 = float(x_slot_angle_157 - x3_box_angle_157)
                                        if bool(globals().get("AUTO_JOINT1_USE_SHORTEST_DELTA_157", True)):
                                            raw_auto_delta_157 = float((raw_auto_delta_157 + np.pi) % (2.0 * np.pi) - np.pi)

                                        # 159_: slot_01과 slot_02가 같은 방향으로 돌면 먼저 놓은 상자를 지나갈 수 있다.
                                        # 따라서 marker index별 선호 회전 방향을 강제한다.
                                        route_sign_159 = 0.0
                                        route_note_159 = "free"
                                        if bool(globals().get("SLOT_ROUTE_SIGN_FORCE_159", False)):
                                            try:
                                                route_pattern_159 = tuple(float(x) for x in globals().get("SLOT_ROUTE_SIGN_BY_MARKER_159", (-1.0, +1.0)))
                                                marker_count_159 = max(1, len(tuple(globals().get("SLOT_MARKER_PATHS_155", ("/World/APalt_slot_01", "/World/APalt_slot_02")))))
                                                marker_idx_159 = int(phase_info.get("stack_slot_index_145", 0)) % marker_count_159
                                                route_sign_159 = float(route_pattern_159[marker_idx_159 % len(route_pattern_159)]) if route_pattern_159 else 0.0
                                                if route_sign_159 > 0.0 and raw_auto_delta_157 < 0.0:
                                                    raw_auto_delta_157 = float(raw_auto_delta_157 + 2.0 * np.pi)
                                                    route_note_159 = f"force_positive:marker_idx={marker_idx_159}"
                                                elif route_sign_159 < 0.0 and raw_auto_delta_157 > 0.0:
                                                    raw_auto_delta_157 = float(raw_auto_delta_157 - 2.0 * np.pi)
                                                    route_note_159 = f"force_negative:marker_idx={marker_idx_159}"
                                                else:
                                                    route_note_159 = f"keep_sign:marker_idx={marker_idx_159},sign={route_sign_159:+.0f}"
                                            except Exception as _route_exc_159:
                                                route_note_159 = f"route_sign_failed:{_route_exc_159}"

                                        # 기존 max_rad와 별도 159/157 안전 제한 중 더 작은 값을 사용한다.
                                        max_auto_rad_157 = abs(math.radians(float(globals().get("AUTO_JOINT1_MAX_ABS_DELTA_DEG_159", globals().get("AUTO_JOINT1_MAX_ABS_DELTA_DEG_157", 170.0)))))
                                        max_use_rad_157 = min(float(max_rad), float(max_auto_rad_157))
                                        auto_delta_157 = float(sign) * float(raw_auto_delta_157)
                                        auto_delta_157 = float(np.clip(auto_delta_157, -max_use_rad_157, max_use_rad_157))
                                        min_auto_rad_157 = abs(math.radians(float(globals().get("AUTO_JOINT1_MIN_DELTA_DEG_157", 0.0))))
                                        if min_auto_rad_157 > 0.0 and abs(auto_delta_157) < min_auto_rad_157:
                                            auto_delta_157 = 0.0
                                        auto_delta_157 = _clamp_first_joint_delta(auto_delta_157)
                                        joint1_delta = float(auto_delta_157)
                                        target_j1 = float(ref_j1 + joint1_delta)
                                        signed_delta = float(joint1_delta)
                                        delta_reason = (
                                            f"auto_slot_angle_157:{center_path_now}:"
                                            f"box_angle={math.degrees(x3_box_angle_157):+.1f}deg,"
                                            f"slot_angle={math.degrees(x_slot_angle_157):+.1f}deg,"
                                            f"raw_delta={math.degrees(raw_auto_delta_157):+.1f}deg,"
                                            f"route={route_note_159}"
                                        )
                                        initial_j1 = None
                                        absolute_target_j1 = None
                                        auto_slot_used_157 = True
                                        if bool(globals().get("AUTO_JOINT1_LOG_157", True)):
                                            print(
                                                f"[AUTO_JOINT1_SLOT_157] start. robot_center=({rc157[0]:+.3f},{rc157[1]:+.3f}), "
                                                f"box_xy=({box_center_now[0]:+.3f},{box_center_now[1]:+.3f}), "
                                                f"slot_xy=({place_center_now[0]:+.3f},{place_center_now[1]:+.3f}), "
                                                f"x3_box_angle={math.degrees(x3_box_angle_157):+.1f}deg, "
                                                f"x_slot_angle={math.degrees(x_slot_angle_157):+.1f}deg, "
                                                f"delta_from_current={math.degrees(joint1_delta):+.1f}deg, "
                                                f"current_j1={math.degrees(ref_j1):+.1f}deg, "
                                                f"target_j1={math.degrees(target_j1):+.1f}deg, "
                                                f"limit={math.degrees(max_use_rad_157):.1f}deg, route={route_note_159}. "
                                                f"final placement is APalt_slot correction."
                                            )
                                except Exception as exc:
                                    print(f"[AUTO_JOINT1_SLOT_157][WARN] failed; fallback to previous mode: {exc}")

                            if not auto_slot_used_157:
                                if bool(globals().get("ABSOLUTE_JOINT1_FROM_INITIAL_ENABLED_156", False)):
                                    init_j = globals().get("ROBOT_INITIAL_JOINTS_156", None)
                                    if init_j is not None and len(init_j) > 0:
                                        initial_j1 = float(init_j[0])
                                    else:
                                        # RESET_ROBOT_TO_ZERO=True인 현재 프로젝트 기준 fallback.
                                        initial_j1 = 0.0
                                    # 여기서 signed_delta는 "초기 기준 목표각"이다.
                                    # 현재 joint_1에서 이 각도를 더하는 것이 아니라, 최종 목표를 initial_j1 + signed_delta로 고정한다.
                                    signed_delta = float(np.clip(signed_delta, -max_rad, max_rad))
                                    signed_delta = _clamp_first_joint_delta(signed_delta)
                                    absolute_target_j1 = float(initial_j1 + signed_delta)
                                    target_j1 = absolute_target_j1
                                    joint1_delta = float(target_j1 - ref_j1)
                                    delta_reason = f"abs_initial_156:{delta_reason}"
                                else:
                                    joint1_delta = float(np.clip(signed_delta, -max_rad, max_rad))
                                    joint1_delta = _clamp_first_joint_delta(joint1_delta)
                                    target_j1 = ref_j1 + joint1_delta

                            if len(target_j) > 0:
                                target_j[0] = target_j1

                            phase_info["start_joints"] = start_j
                            phase_info["target_joints"] = target_j
                            phase_info["ref_j1"] = ref_j1
                            phase_info["initial_j1_156"] = initial_j1
                            phase_info["absolute_target_j1_156"] = absolute_target_j1
                            phase_info["target_j1"] = target_j1
                            phase_info["joint1_delta"] = joint1_delta
                            phase_info["delta_reason"] = delta_reason
                            phase_info["box_center_start"] = box_center_now.copy()
                            phase_info["place_center"] = place_center_now.copy()
                            if bool(globals().get("ABSOLUTE_JOINT1_FROM_INITIAL_ENABLED_156", False)) and absolute_target_j1 is not None:
                                init_log = float(initial_j1) if initial_j1 is not None else 0.0
                                abs_log = float(absolute_target_j1) if absolute_target_j1 is not None else float(target_j1)
                                print(
                                    f"[ABS_JOINT1_PLACE_156] start. initial_j1={init_log:+.6f}rad/{math.degrees(init_log):+.1f}deg, "
                                    f"current_ref_j1={ref_j1:+.6f}rad/{math.degrees(ref_j1):+.1f}deg, "
                                    f"target_abs_j1={abs_log:+.6f}rad/{math.degrees(abs_log):+.1f}deg, "
                                    f"delta_from_current={joint1_delta:+.6f}rad/{math.degrees(joint1_delta):+.1f}deg, "
                                    f"reason={delta_reason}, only joint_1 changes. "
                                    f"box_xy=({box_center_now[0]:.3f},{box_center_now[1]:.3f}), "
                                    f"place_xy=({place_center_now[0]:.3f},{place_center_now[1]:.3f})"
                                )
                            else:
                                print(
                                    f"[SIMPLE_STEP2_JOINT1_ROTATE_117] start. ref_j1={ref_j1:+.6f}, "
                                    f"target_j1={target_j1:+.6f}, j1_delta={joint1_delta:+.6f}rad/{math.degrees(joint1_delta):+.1f}deg, "
                                    f"max={float(phase_info.get('max_deg', HYBRID_JOINT1_ROTATE_MAX_DEG)):.1f}deg, reason={delta_reason}, only joint_1 changes. "
                                    f"box_xy=({box_center_now[0]:.3f},{box_center_now[1]:.3f}), "
                                    f"place_xy=({place_center_now[0]:.3f},{place_center_now[1]:.3f})"
                                )
                        start_j = np.array(phase_info["start_joints"], dtype=float)
                        target_j = np.array(phase_info["target_joints"], dtype=float)
                        alpha_raw = float(np.clip(custom_phase_step / float(duration), 0.0, 1.0))
                        alpha = float(alpha_raw * alpha_raw * (3.0 - 2.0 * alpha_raw))
                        cmd_j = (1.0 - alpha) * start_j + alpha * target_j
                        robot.apply_action(ArticulationAction(joint_positions=cmd_j))
                        suction_after = update_vgc10_suction_anchor(robot)
                        bbox_after = get_world_bbox_info(stage, task.box_path)
                        c_after = bbox_after["center"] if bbox_after is not None else np.zeros(3)
                        root_pos = get_world_translation(stage, task.box_path)
                        root_minus_bbox = float(root_pos[2] - c_after[2]) if root_pos is not None else 999.0
                        ref_j1 = float(phase_info.get("ref_j1", start_j[0] if len(start_j)>0 else 0.0))
                        j1_delta_now = float(cmd_j[0] - ref_j1) if len(cmd_j) > 0 else 0.0
                        init_j1_log_156 = phase_info.get("initial_j1_156", None)
                        j1_from_initial_now_156 = None
                        if init_j1_log_156 is not None and len(cmd_j) > 0:
                            try:
                                j1_from_initial_now_156 = float(cmd_j[0] - float(init_j1_log_156))
                            except Exception:
                                j1_from_initial_now_156 = None
                        phase_done = bool(custom_phase_step >= duration)
                        if custom_phase_step % CUSTOM_LOG_INTERVAL == 0 or phase_done:
                            print(
                                f"  [simple_step2_joint1_rotate_117] step={custom_phase_step}/{duration}, alpha={alpha_raw:.2f}, "
                                f"j1={cmd_j[0] if len(cmd_j)>0 else 0.0:+.3f}, "
                                f"j1_delta_from_attach={j1_delta_now:+.3f}rad/{math.degrees(j1_delta_now):+.1f}deg, "
                                f"j1_from_initial={(j1_from_initial_now_156 if j1_from_initial_now_156 is not None else 0.0):+.3f}rad/{(math.degrees(j1_from_initial_now_156) if j1_from_initial_now_156 is not None else 0.0):+.1f}deg, "
                                f"j2={cmd_j[1] if len(cmd_j)>1 else 0.0:+.3f}, "
                                f"j3={cmd_j[2] if len(cmd_j)>2 else 0.0:+.3f}, "
                                f"suction=({suction_after[0]:.3f},{suction_after[1]:.3f},{suction_after[2]:.3f}), "
                                f"box=({c_after[0]:.3f},{c_after[1]:.3f},{c_after[2]:.4f}), "
                                f"upright_check_117=root_minus_bbox_z={root_minus_bbox:+.4f}"
                            )
                            log_oribox_pose_tracker(stage, f"simple_joint1_rotate_116:{custom_phase_step}", force=True)
                        custom_phase_step += 1
                        if phase_done:
                            if custom_phase_index < len(phase_sequence) - 1:
                                custom_phase_index += 1
                                custom_phase_step = 0
                                cart_controller.reset()

                                # 158_: joint_1 회전이 끝난 직후 다음 Cartesian phase의 start를
                                #       회전 전 lift_suction이 아니라 실제 현재 suction 위치로 교체한다.
                                #       이게 없으면 회전 후 APalt_slot 근처까지 갔다가 다시 컨베이어 쪽
                                #       lift 위치로 되돌아가려는 이상한 움직임이 생긴다.
                                next_phase_158 = phase_sequence[custom_phase_index]
                                next_name_158 = str(next_phase_158.get("name", ""))
                                if (
                                    bool(globals().get("PHASE_DYNAMIC_START_AFTER_JOINT1_ENABLED_158", True))
                                    and next_name_158 in tuple(globals().get("PHASE_DYNAMIC_START_NAMES_158", ()))
                                ):
                                    old_start_158 = np.array(next_phase_158.get("start", suction_after), dtype=float).copy()
                                    actual_start_158 = np.array(suction_after, dtype=float).copy()
                                    next_phase_158["start"] = actual_start_158.copy()

                                    # 160_: 회전 후 이미 slot 근처인데 move_over가 target z를 더 높게 잡으면
                                    #       release 직전에 살짝 들어올리는 동작이 생긴다. move_over에서는 z를 올리지 않는다.
                                    no_up_note_160 = ""
                                    if bool(globals().get("SLOT_MARKER_NO_UP_BEFORE_RELEASE_160", True)) and next_name_158 == "slot_marker_move_over_155":
                                        try:
                                            target_160 = np.array(next_phase_158.get("target", actual_start_158), dtype=float).copy()
                                            old_target_160 = target_160.copy()
                                            if float(target_160[2]) > float(actual_start_158[2]):
                                                target_160[2] = float(actual_start_158[2])
                                                next_phase_158["target"] = target_160.copy()
                                                no_up_note_160 += f", no_up_z:{old_target_160[2]:.3f}->{target_160[2]:.3f}"
                                            # XY가 이미 충분히 가까우면 move_over 시간을 거의 없앤다.
                                            xy_err_160 = float(np.linalg.norm(target_160[:2] - actual_start_158[:2]))
                                            if bool(globals().get("SLOT_MARKER_SKIP_MOVE_IF_NEAR_XY_160", True)) and xy_err_160 <= float(globals().get("SLOT_MARKER_SKIP_MOVE_XY_TOL_160", 0.08)):
                                                old_steps_160 = int(next_phase_158.get("steps", 1))
                                                next_phase_158["steps"] = int(min(old_steps_160, 25))
                                                no_up_note_160 += f", near_xy={xy_err_160:.4f}, steps:{old_steps_160}->{next_phase_158['steps']}"
                                        except Exception as _fix160_exc:
                                            no_up_note_160 += f", no_up_fix_warn={type(_fix160_exc).__name__}"

                                    next_phase_158["dynamic_start_set_158"] = True
                                    next_phase_158["dynamic_start_reason_158"] = "after_joint1_rotate"
                                    if bool(globals().get("PHASE_DYNAMIC_START_LOG_158", True)):
                                        _t158 = np.array(next_phase_158.get('target', actual_start_158), dtype=float)
                                        print(
                                            f"[PHASE_START_FIX_158] next={next_name_158}, reason=after_joint1_rotate, "
                                            f"old_start=({old_start_158[0]:.3f},{old_start_158[1]:.3f},{old_start_158[2]:.3f}) -> "
                                            f"actual_start=({actual_start_158[0]:.3f},{actual_start_158[1]:.3f},{actual_start_158[2]:.3f}), "
                                            f"target=({_t158[0]:.3f},{_t158[1]:.3f},{_t158[2]:.3f}){no_up_note_160}"
                                        )

                                print(f"[SIMPLE_CARRY_116] phase done -> next={phase_sequence[custom_phase_index].get('name')}")
                            else:
                                custom_carry_active = False
                        was_playing = is_playing
                        continue

                    # legacy 114_: joint_2 회전 블록. 117_ 기본 흐름에서는 사용하지 않는다.
                    if phase_kind == "joint2_rotate":
                        duration = int(max(1, phase_info.get("steps", HYBRID_LINK2_ROTATE_STEPS)))
                        if "start_joints" not in phase_info:
                            start_j = np.array(robot.get_joint_positions(), dtype=float)
                            target_j = start_j.copy()
                            ref_j2 = float(start_j[1]) if len(start_j) > 1 else 0.0
                            target_link2_z = float(phase_info.get("target_link2_z_deg", HYBRID_LINK2_ROTATE_TARGET_DEG))
                            # 기준식: link2_z_est = -90 + degrees(j2 - ref_j2)
                            # target_j2 = ref_j2 + radians(target_link2_z - (-90))
                            target_j2 = ref_j2 + math.radians(target_link2_z - (-90.0))
                            min_link2_z = float(phase_info.get("min_link2_z_deg", HYBRID_LINK2_ROTATE_MIN_DEG))
                            min_j2 = ref_j2 + math.radians(min_link2_z - (-90.0))
                            # 너무 과하게 접히지 않게 하한 guard. target_j2가 min보다 더 작으면 min으로 제한.
                            if target_j2 < min_j2:
                                target_j2 = min_j2
                            if len(target_j) > 1:
                                target_j[1] = target_j2
                            phase_info["start_joints"] = start_j
                            phase_info["target_joints"] = target_j
                            phase_info["ref_j2"] = ref_j2
                            phase_info["target_j2"] = target_j2
                            print(
                                f"[SIMPLE_STEP2_LINK2_ROTATE_114] start. ref_j2={ref_j2:+.6f}, "
                                f"target_link2_z={target_link2_z:+.1f}deg, min_link2_z={min_link2_z:+.1f}deg, "
                                f"target_j2={target_j2:+.6f}; only joint_2 changes."
                            )
                        start_j = np.array(phase_info["start_joints"], dtype=float)
                        target_j = np.array(phase_info["target_joints"], dtype=float)
                        alpha_raw = float(np.clip(custom_phase_step / float(duration), 0.0, 1.0))
                        alpha = float(alpha_raw * alpha_raw * (3.0 - 2.0 * alpha_raw))
                        cmd_j = (1.0 - alpha) * start_j + alpha * target_j
                        robot.apply_action(ArticulationAction(joint_positions=cmd_j))
                        suction_after = update_vgc10_suction_anchor(robot)
                        bbox_after = get_world_bbox_info(stage, task.box_path)
                        c_after = bbox_after["center"] if bbox_after is not None else np.zeros(3)
                        root_pos = get_world_translation(stage, task.box_path)
                        root_minus_bbox = float(root_pos[2] - c_after[2]) if root_pos is not None else 999.0
                        ref_j2 = float(phase_info.get("ref_j2", start_j[1] if len(start_j)>1 else 0.0))
                        link2_z_est = -90.0 + math.degrees(float(cmd_j[1] - ref_j2)) if len(cmd_j) > 1 else -90.0
                        phase_done = bool(custom_phase_step >= duration)
                        if custom_phase_step % CUSTOM_LOG_INTERVAL == 0 or phase_done:
                            print(
                                f"  [simple_step2_link2_rotate_114] step={custom_phase_step}/{duration}, alpha={alpha_raw:.2f}, "
                                f"j1={cmd_j[0] if len(cmd_j)>0 else 0.0:+.3f}, "
                                f"j2={cmd_j[1] if len(cmd_j)>1 else 0.0:+.3f}, "
                                f"j3={cmd_j[2] if len(cmd_j)>2 else 0.0:+.3f}, "
                                f"link2_z_est={link2_z_est:+.2f}deg, "
                                f"suction=({suction_after[0]:.3f},{suction_after[1]:.3f},{suction_after[2]:.3f}), "
                                f"box=({c_after[0]:.3f},{c_after[1]:.3f},{c_after[2]:.4f}), "
                                f"upright_check_117=root_minus_bbox_z={root_minus_bbox:+.4f}"
                            )
                            log_oribox_pose_tracker(stage, f"hybrid_link2_rotate_111:{custom_phase_step}", force=True)
                        custom_phase_step += 1
                        if phase_done:
                            if custom_phase_index < len(phase_sequence) - 1:
                                custom_phase_index += 1
                                custom_phase_step = 0
                                cart_controller.reset()
                                print(f"[HYBRID_CARRY_111] phase done -> next={phase_sequence[custom_phase_index].get('name')}")
                            else:
                                custom_carry_active = False
                        was_playing = is_playing
                        continue

                    # 111_: joint_2 회전이 끝난 현재 위치에서, 처음 vertical_lift의 반대 방향으로 수직 하강한다.
                    if phase_kind == "reverse_vertical_lower" and "target" not in phase_info:
                        start_suction = np.array(suction_now, dtype=float)
                        target_suction = start_suction.copy()
                        target_suction[2] = float(start_suction[2]) + float(phase_info.get("delta_z", -float(VERTICAL_LIFT_DELTA_Z)))
                        phase_info["start"] = start_suction.copy()
                        phase_info["target"] = target_suction.copy()
                        print(
                            f"[SIMPLE_STEP3_REVERSE_LOWER_117] dynamic target set. "
                            f"start=({start_suction[0]:.3f},{start_suction[1]:.3f},{start_suction[2]:.3f}), "
                            f"target=({target_suction[0]:.3f},{target_suction[1]:.3f},{target_suction[2]:.3f})"
                        )

                    # 111_: settle phase도 직전 실제 suction 위치를 유지하도록 동적으로 설정한다.
                    if phase_kind == "path_settle" and "target" not in phase_info:
                        phase_info["start"] = np.array(suction_now, dtype=float).copy()
                        phase_info["target"] = np.array(suction_now, dtype=float).copy()

                    # 171_: release 직전 APalt_slot 방향 기준 yaw만 RMPFlow target orientation으로 보정한다.
                    # target suction position은 현재 위치로 고정해서 중심/수평을 건드리지 않는 것을 우선한다.
                    if phase_kind == "pre_release_yaw_align_171":
                        _prepare_pre_release_yaw_align_phase_171(stage, robot, task, phase_info, suction_now)

                    # 184_: 2번째 상자 yaw 정렬을 별도 pre_release phase로 하지 않고,
                    # slot_marker_lower_155 / slot_marker_settle_155가 수행되는 동안 target orientation에 섞는다.
                    if bool(phase_info.get("fused_yaw_align_184", False)):
                        _prepare_fused_yaw_align_phase_184(stage, robot, task, phase_info, suction_now)

                    # 158_ 2차 안전장치:
                    # phase 전환 시점에 start를 갱신하지 못했거나 RMPFlow/PhysX 때문에 실제 suction 위치가
                    # 계획 위치와 달라진 경우에도, slot_marker_* phase의 첫 step에서는 반드시 현재 실제 위치에서
                    # 보간을 시작한다. 이렇게 해야 step=0 target_suction이 회전 전 lift_suction으로 되돌아가지 않는다.
                    if (
                        bool(globals().get("PHASE_DYNAMIC_START_AFTER_JOINT1_ENABLED_158", True))
                        and custom_phase_step == 0
                        and str(phase_info.get("name", "")) in tuple(globals().get("PHASE_DYNAMIC_START_NAMES_158", ()))
                        and "target" in phase_info
                    ):
                        actual_start_158b = np.array(suction_now, dtype=float).copy()
                        old_start_158b = np.array(phase_info.get("start", actual_start_158b), dtype=float).copy()
                        phase_info["start"] = actual_start_158b.copy()
                        no_up_note_160b = ""
                        if bool(globals().get("SLOT_MARKER_NO_UP_BEFORE_RELEASE_160", True)) and str(phase_info.get("name", "")) == "slot_marker_move_over_155":
                            try:
                                target_160b = np.array(phase_info.get("target", actual_start_158b), dtype=float).copy()
                                old_target_160b = target_160b.copy()
                                if float(target_160b[2]) > float(actual_start_158b[2]):
                                    target_160b[2] = float(actual_start_158b[2])
                                    phase_info["target"] = target_160b.copy()
                                    no_up_note_160b += f", no_up_z:{old_target_160b[2]:.3f}->{target_160b[2]:.3f}"
                                xy_err_160b = float(np.linalg.norm(target_160b[:2] - actual_start_158b[:2]))
                                if bool(globals().get("SLOT_MARKER_SKIP_MOVE_IF_NEAR_XY_160", True)) and xy_err_160b <= float(globals().get("SLOT_MARKER_SKIP_MOVE_XY_TOL_160", 0.08)):
                                    old_steps_160b = int(phase_info.get("steps", 1))
                                    phase_info["steps"] = int(min(old_steps_160b, 25))
                                    no_up_note_160b += f", near_xy={xy_err_160b:.4f}, steps:{old_steps_160b}->{phase_info['steps']}"
                            except Exception as _fix160b_exc:
                                no_up_note_160b += f", no_up_runtime_warn={type(_fix160b_exc).__name__}"
                        phase_info["dynamic_start_runtime_set_158"] = True
                        if bool(globals().get("PHASE_DYNAMIC_START_LOG_158", True)):
                            target_dbg_158b = np.array(phase_info.get("target", actual_start_158b), dtype=float).copy()
                            print(
                                f"[PHASE_START_RUNTIME_FIX_158] phase={phase_info.get('name')}, "
                                f"old_start=({old_start_158b[0]:.3f},{old_start_158b[1]:.3f},{old_start_158b[2]:.3f}) -> "
                                f"actual_start=({actual_start_158b[0]:.3f},{actual_start_158b[1]:.3f},{actual_start_158b[2]:.3f}), "
                                f"target=({target_dbg_158b[0]:.3f},{target_dbg_158b[1]:.3f},{target_dbg_158b[2]:.3f}){no_up_note_160b}"
                            )

                    target_final = np.array(phase_info["target"], dtype=float)
                    target_start = np.array(phase_info.get("start", target_final), dtype=float)
                    phase_steps = int(max(1, phase_info.get("steps", CUSTOM_PHASE_MAX_STEPS)))
                    alpha_raw = float(np.clip(custom_phase_step / float(phase_steps), 0.0, 1.0))
                    # 부드러운 S-curve 보간. target_suction이 갑자기 크게 변하지 않게 한다.
                    alpha = float(alpha_raw * alpha_raw * (3.0 - 2.0 * alpha_raw))
                    target_suction = target_start + (target_final - target_start) * alpha
                    phase_error = float(np.linalg.norm(np.array(suction_now, dtype=float) - target_suction))
                    phase_done = bool(custom_phase_step >= phase_steps)

                    if phase_kind in ("path_move", "path_lower", "path_lift", "path_settle", "reverse_vertical_lower"):
                        # 59_: target_suction을 현재 위치/guard 기준으로 바꾸지 않는다.
                        # 56_ 로그에서 target_suction이 중간중간 엉뚱한 좌표로 튄 원인을 제거한다.
                        pass
                    elif phase_kind == "lift":
                        phase_error = float(abs(float(suction_now[2]) - float(target_suction[2])))
                        phase_done = (phase_error <= CUSTOM_LIFT_TOL) or (custom_phase_step >= CUSTOM_PHASE_MAX_STEPS)
                    elif phase_kind == "lower":
                        phase_error = float(np.linalg.norm(np.array(suction_now, dtype=float) - target_suction))
                        phase_done = (
                            (phase_error <= CUSTOM_LOWER_TOL and custom_phase_step >= CUSTOM_RELEASE_AFTER_LOWER_MIN_STEPS)
                            or (custom_phase_step >= CUSTOM_PHASE_MAX_STEPS)
                        )
                    else:
                        phase_error = float(np.linalg.norm(np.array(suction_now[:2], dtype=float) - target_suction[:2]))
                        phase_done = (phase_error <= CUSTOM_MOVE_XY_TOL) or (custom_phase_step >= CUSTOM_PHASE_MAX_STEPS)

                    # 173_: pre_release_yaw_align_171은 target_suction을 현재 위치에 고정한 채
                    # orientation만 RMPFlow로 먹이는 phase다. 171 실패 원인은 target 위치 오차가 0이라
                    # step=0에서 phase_done=True가 되어 실제 action이 여러 step 적용되지 않은 것.
                    # 172 probe에서 검증된 것처럼 정해진 step 수 동안 반드시 action을 넣는다.
                    if phase_kind == "pre_release_yaw_align_171":
                        target_suction = np.array(phase_info.get("hold_suction_171", suction_now), dtype=float).copy()
                        if bool(phase_info.get("skip_171", False)):
                            phase_error = 0.0
                            phase_done = True
                        elif bool(phase_info.get("abort_171", False)):
                            phase_done = True
                        else:
                            phase_error = 999.0  # yaw axis error는 아래 진단 로그에서 별도로 출력
                            phase_done = bool(custom_phase_step >= int(max(1, phase_info.get("steps", PRE_RELEASE_YAW_ALIGN_STEPS_171))))

                    # 24_: min_center_z를 쓰면 suction이 실제로 못 올라갔는데 박스만 높게 튀는 순간이동이 생긴다.
                    # 그래서 박스는 항상 실제 suction_after + attach offset만 따라간다.
                    min_center_z = None
                else:
                    if custom_carry_phase == "lift":
                        target_suction = np.array(custom_targets["lift_suction"], dtype=float)
                        phase_error = float(np.linalg.norm(np.array(suction_now, dtype=float) - target_suction))
                        phase_done = (abs(float(suction_now[2]) - float(target_suction[2])) <= CUSTOM_LIFT_TOL) or (custom_phase_step >= CUSTOM_PHASE_MAX_STEPS)
                    elif custom_carry_phase == "move":
                        target_suction = np.array(custom_targets["move_suction"], dtype=float)
                        phase_error = float(np.linalg.norm(np.array(suction_now[:2], dtype=float) - target_suction[:2]))
                        phase_done = (phase_error <= CUSTOM_MOVE_XY_TOL) or (custom_phase_step >= CUSTOM_PHASE_MAX_STEPS)
                    elif custom_carry_phase == "lower":
                        target_suction = np.array(custom_targets["lower_suction"], dtype=float)
                        phase_error = float(np.linalg.norm(np.array(suction_now, dtype=float) - target_suction))
                        phase_done = (phase_error <= CUSTOM_LOWER_TOL and custom_phase_step >= CUSTOM_RELEASE_AFTER_LOWER_MIN_STEPS) or (custom_phase_step >= CUSTOM_PHASE_MAX_STEPS)
                    else:
                        target_suction = np.array(suction_now, dtype=float)
                        phase_error = 0.0
                        phase_done = True
                    min_center_z = None

                # 173_: pre_release_yaw_align_171 phase에서는 172 probe에서 성공한 방식과 동일하게
                # target_end_effector_position은 현재 위치에 가깝게 고정하고, target_end_effector_orientation만
                # APalt_slot yaw 오차로 만든 quaternion으로 넣는다.
                phase_fixed_orientation_171 = phase_info.get("fixed_orientation_171", custom_fixed_orientation) if isinstance(phase_info, dict) else custom_fixed_orientation
                apply_cartesian_suction_target(
                    cart_controller,
                    robot,
                    current_suction_pos=suction_now,
                    desired_suction_pos=target_suction,
                    fixed_orientation=phase_fixed_orientation_171,
                )

                suction_after = update_vgc10_suction_anchor(robot)
                if PHYSICS_FIXED_JOINT_ATTACH_ENABLED:
                    # 51_: 실제 물리 joint가 박스를 끌고 가게 둔다. 박스 transform 직접 이동 금지.
                    bbox_tmp = get_world_bbox_info(stage, task.box_path)
                    desired_center = bbox_tmp["center"] if bbox_tmp is not None else np.zeros(3)
                    ok_follow = True
                else:
                    if BOX_DISABLE_PHYSICS_DURING_CARRY:
                        set_box_scripted_carry_mode(stage, task.box_move_path, True, verbose=False)
                    # 59_: RMPFlow/IK가 순간적으로 흔들려도 박스가 같이 미쳐 날뛰지 않게,
                    # 운반 중 박스는 고정 waypoint의 desired suction path를 따른다.
                    # 로봇은 같은 target을 향해 움직이지만, 박스 적재 안정성을 우선한다.
                    follow_suction_for_box = target_suction if TOP_LOCK_FOLLOW_DESIRED_PATH else suction_after
                    ok_follow, desired_center = follow_box_to_suction(
                        stage,
                        task,
                        suction_pos=follow_suction_for_box,
                        attached_center_offset=attached_center_offset,
                        min_center_z=min_center_z,
                    )

                if custom_phase_step % CUSTOM_LOG_INTERVAL == 0 or phase_done:
                    bbox_after = get_world_bbox_info(stage, task.box_path)
                    c_after = bbox_after["center"] if bbox_after is not None else desired_center

                    # 110_ bug fix:
                    # 109_에서 get_world_transform()이라는 없는 함수를 호출해서 NameError가 났다.
                    # 기존 파일에 이미 정의되어 있는 get_world_translation()으로 root 위치를 읽는다.
                    box_root_pos_for_upright = get_world_translation(stage, task.box_path)
                    if box_root_pos_for_upright is not None and c_after is not None:
                        root_minus_bbox_z_110 = float(box_root_pos_for_upright[2] - c_after[2])
                    else:
                        root_minus_bbox_z_110 = 999.0

                    print(
                        f"  [custom_{custom_carry_phase}] step={custom_phase_step}, "
                        f"err={phase_error:.4f}, "
                        f"target_suction=({target_suction[0]:.3f},{target_suction[1]:.3f},{target_suction[2]:.3f}), "
                        f"suction=({suction_after[0]:.3f},{suction_after[1]:.3f},{suction_after[2]:.3f}), "
                        f"box=({c_after[0]:.3f},{c_after[1]:.3f},{c_after[2]:.4f}), "
                        f"upright_check_117=root_minus_bbox_z={root_minus_bbox_z_110:+.4f}, "
                        f"follow_ok={ok_follow}, top_lock_path={TOP_LOCK_FOLLOW_DESIRED_PATH}"
                    )
                    log_oribox_pose_tracker(stage, f"custom_{custom_carry_phase}:{custom_phase_step}", force=True)

                    if phase_kind == "pre_release_yaw_align_171":
                        try:
                            _slot_idx_171_dbg = int(getattr(task, "_stack_slot_index", 0))
                            _diag171_now = apalt_release_pose_diagnose_only_170(stage, task.box_move_path, _slot_idx_171_dbg, label=f"during_pre_release_yaw_align_173:{custom_phase_step}")
                            if isinstance(_diag171_now, dict):
                                _c_err171 = float(_diag171_now.get("center_xy_err", 999.0))
                                _lvl171 = float(_diag171_now.get("box_to_slot_z_deg", 999.0))
                                _axis171 = float(_diag171_now.get("best_axis_err", 999.0))
                                print(
                                    f"[PRE_RELEASE_YAW_ALIGN_173][CHECK] step={custom_phase_step}, "
                                    f"center_xy_err={_c_err171:.4f}, level={_lvl171:.2f}deg, axis_err={_axis171:.2f}deg"
                                )
                                if _c_err171 > float(globals().get("PRE_RELEASE_YAW_ALIGN_CENTER_ABORT_M_171", 0.018)) or _lvl171 > float(globals().get("PRE_RELEASE_YAW_ALIGN_LEVEL_ABORT_DEG_171", 3.0)):
                                    phase_info["abort_171"] = True
                                    phase_info["abort_reason_171"] = f"center_or_level_drift:center={_c_err171:.4f},level={_lvl171:.2f}"
                                    phase_done = True
                                    print(f"[PRE_RELEASE_YAW_ALIGN_173][ABORT] {phase_info['abort_reason_171']}")
                        except Exception as _align171_log_exc:
                            print(f"[PRE_RELEASE_YAW_ALIGN_173][WARN] check failed: {type(_align171_log_exc).__name__}: {_align171_log_exc}")

                custom_phase_step += 1

                if phase_done:
                    phase_sequence = custom_targets.get("phase_sequence") if isinstance(custom_targets, dict) else None

                    # 67_ 핵심 수정:
                    # path_move/path_lower/path_settle 단계가 단순 step timeout으로 끝나면,
                    # 로봇 suction이 BoxAprop 목표 근처에 실제로 도달하지 못했는데도 다음 lower/release 단계로 넘어간다.
                    # 그 결과 두 번째 박스가 현재 위치에 남은 채 [BOXAPROP_RELEASE_ABORT]로 pause된다.
                    # 목표 오차가 큰 경우에는 더 진행하지 않고 release 없이 home으로 복귀해서 멈춤/순간이동을 막는다.
                    if (
                        phase_sequence
                        and bool(custom_targets.get("boxaprop_place_enabled", custom_targets.get("cube_over_enabled", False)))
                        and bool(BOXAPROP_ABORT_HOME_ON_TARGET_ERROR)
                        and custom_carry_phase in ("boxaprop_move", "cube_over_move", "boxaprop_lower", "boxaprop_settle")
                    ):
                        max_err = float(BOXAPROP_MOVE_MAX_SUCTION_ERR) if custom_carry_phase in ("boxaprop_move", "cube_over_move") else float(BOXAPROP_LOWER_MAX_SUCTION_ERR)
                        if float(phase_error) > max_err:
                            bbox_abort = get_world_bbox_info(stage, task.box_path)
                            c_abort = bbox_abort["center"] if bbox_abort is not None else np.zeros(3)
                            print(
                                f"[BOXAPROP_MOVE_ABORT] phase={custom_carry_phase}, "
                                f"target_err={phase_error:.3f}m > {max_err:.3f}m. "
                                f"release/lower를 진행하지 않고 home 복귀합니다. "
                                f"box=({c_abort[0]:.3f},{c_abort[1]:.3f},{c_abort[2]:.4f}), "
                                f"target_suction=({target_suction[0]:.3f},{target_suction[1]:.3f},{target_suction[2]:.3f}), "
                                f"actual_suction=({suction_after[0]:.3f},{suction_after[1]:.3f},{suction_after[2]:.3f})"
                            )
                            log_oribox_pose_tracker(stage, f"boxaprop_move_abort:{custom_carry_phase}", force=True)
                            attached = False
                            custom_carry_active = False
                            custom_carry_phase = None
                            attached_center_offset = None
                            attach_center_z = None
                            start_home_return("boxaprop_move_abort_target_error", success=False)
                            was_playing = is_playing
                            continue

                    if phase_sequence and custom_phase_index < len(phase_sequence) - 1:
                        custom_phase_index += 1
                        custom_carry_phase = phase_sequence[custom_phase_index].get("name", str(custom_phase_index))
                        custom_phase_step = 0
                        cart_controller.reset()
                        # 112_ bug fix:
                        # 111_ hybrid phases such as link2_rotate_to_back do not have a Cartesian
                        # "target" key.  After vertical_lift the old transition code tried to
                        # print phase_sequence[...]["target"], causing KeyError: 'target'.
                        next_phase_info_112 = phase_sequence[custom_phase_index]
                        if "target" in next_phase_info_112:
                            next_target = np.array(next_phase_info_112["target"], dtype=float)
                            print(
                                f"[CUSTOM_CARRY_116] phase done -> next={custom_carry_phase}, "
                                f"target=({next_target[0]:.3f},{next_target[1]:.3f},{next_target[2]:.3f})"
                            )
                        else:
                            print(
                                f"[CUSTOM_CARRY_116] phase done -> next={custom_carry_phase}, "
                                f"kind={next_phase_info_112.get('kind', 'unknown')}, no Cartesian target; dynamic/joint phase."
                            )
                    elif (not phase_sequence) and custom_carry_phase == "lift":
                        custom_carry_phase = "move"
                        custom_phase_step = 0
                        cart_controller.reset()
                        print("[CUSTOM_CARRY] lift done -> move across robot center at safe height")
                    elif (not phase_sequence) and custom_carry_phase == "move":
                        custom_carry_phase = "lower"
                        custom_phase_step = 0
                        cart_controller.reset()
                        print("[CUSTOM_CARRY] move done -> lower at opposite coordinate")
                    else:
                        if bool(VERTICAL_LIFT_ONLY_TEST) and isinstance(custom_targets, dict) and custom_targets.get("vertical_only", False):
                            bbox_after = get_world_bbox_info(stage, task.box_path)
                            c_after = bbox_after["center"] if bbox_after is not None else np.zeros(3)
                            suction_final = update_vgc10_suction_anchor(robot)
                            place_enabled = bool(custom_targets.get("boxaprop_place_enabled", custom_targets.get("cube_over_enabled", False)))
                            release_enabled = bool(custom_targets.get("boxaprop_release", False))
                            force_drop_release_126_dbg = (
                                str(CUSTOM_CARRY_MODE) == "VERTICAL_JOINT1_REVERSE"
                                and bool(globals().get("FORCE_DROP_RELEASE_AFTER_SETTLE_126", True))
                            )
                            lower_target = np.array(custom_targets.get("lower_suction", suction_final), dtype=float)
                            place_center = np.array(custom_targets.get("place_center", c_after), dtype=float)
                            final_err = float(np.linalg.norm(np.array(suction_final, dtype=float) - lower_target)) if place_enabled else 0.0
                            center_err = float(np.linalg.norm(np.array(c_after[:2], dtype=float) - place_center[:2])) if place_enabled else 0.0
                            rr_place = robot_relative_vector(stage, place_center)
                            print(
                                f"[UPRIGHT_RMPFLOW_DONE_109] box=({c_after[0]:.3f},{c_after[1]:.3f},{c_after[2]:.4f}), "
                                f"suction=({suction_final[0]:.3f},{suction_final[1]:.3f},{suction_final[2]:.3f}), "
                                f"lower_target=({lower_target[0]:.3f},{lower_target[1]:.3f},{lower_target[2]:.3f}), "
                                f"target_err={final_err:.4f}, center_xy_err={center_err:.4f}, release_enabled={release_enabled}, force_drop_release_126={force_drop_release_126_dbg}, "
                                f"place_center_world=({place_center[0]:.4f}, {place_center[1]:.4f}, {place_center[2]:.4f}), "
                                f"place_center_robot_relative=({rr_place[0]:+.4f}, {rr_place[1]:+.4f}, {rr_place[2]:+.4f})"
                            )
                            log_oribox_pose_tracker(stage, "boxaprop_place_done_before_release", force=True)

                            # 171_: release 직전 yaw-align phase가 중심/수평 drift로 중단된 경우에는 release하지 않는다.
                            # 새 문제를 만들지 않기 위해 상자를 떼지 않고 일시정지해서 로그를 확인한다.
                            if isinstance(phase_info, dict) and bool(phase_info.get("abort_171", False)):
                                print(f"[PRE_RELEASE_YAW_ALIGN_173][RELEASE_BLOCKED] {phase_info.get('abort_reason_171', 'abort')}. surface remains ON; release skipped.")
                                if bool(globals().get("PRE_RELEASE_YAW_ALIGN_ABORT_PAUSE_171", True)):
                                    try:
                                        world.pause()
                                    except Exception:
                                        pass
                                was_playing = is_playing
                                continue

                            if bool(globals().get("PRE_RELEASE_YAW_ALIGN_REQUIRE_AXIS_OK_BEFORE_RELEASE_171", False)):
                                try:
                                    _slot_idx_req171 = int(getattr(task, "_stack_slot_index", 0))
                                    _diag_req171 = apalt_release_pose_diagnose_only_170(stage, task.box_move_path, _slot_idx_req171, label="after_pre_release_yaw_align_173_require_check")
                                    if isinstance(_diag_req171, dict) and not bool(_diag_req171.get("ok_axis", False)):
                                        print(f"[PRE_RELEASE_YAW_ALIGN_173][RELEASE_BLOCKED] axis still not OK: {float(_diag_req171.get('best_axis_err', 999.0)):.2f}deg")
                                        try:
                                            world.pause()
                                        except Exception:
                                            pass
                                        was_playing = is_playing
                                        continue
                                except Exception as _req171_exc:
                                    print(f"[PRE_RELEASE_YAW_ALIGN_173][WARN] require check failed: {type(_req171_exc).__name__}: {_req171_exc}")

                            if place_enabled and final_err > float(BOXAPROP_RELEASE_MAX_SUCTION_ERR):
                                print(
                                    f"[BOXAPROP_RELEASE_ABORT] suction target error too large: {final_err:.3f}m > {BOXAPROP_RELEASE_MAX_SUCTION_ERR:.3f}m. "
                                    "release하지 않고 실패 처리 후 home 복귀합니다. BoxAprop 좌표/로봇 도달 범위를 조정해야 합니다."
                                )
                                log_oribox_pose_tracker(stage, "boxaprop_release_abort_before_home", force=True)
                                attached = False
                                custom_carry_active = False
                                custom_carry_phase = None
                                attached_center_offset = None
                                attach_center_z = None
                                start_home_return("boxaprop_release_abort_target_error", success=False)
                                was_playing = is_playing
                                continue

                            # 125_ 핵심 수정:
                            # 117/124의 VERTICAL_JOINT1_REVERSE 경로에서는 VERTICAL_LIFT_THEN_CUBE_OVER_ENABLED=False라
                            # place_enabled가 False로 남는다. 124번은 if place_enabled and release_enabled 조건 때문에
                            # hybrid_settle까지 끝나도 실제 detach 블록에 들어오지 못했다.
                            # 이제 joint_1 회전/하강/settle이 끝나면 place_enabled와 무관하게 FixedJoint를 제거한다.
                            force_drop_release_126 = (
                                str(CUSTOM_CARRY_MODE) == "VERTICAL_JOINT1_REVERSE"
                                and bool(globals().get("FORCE_DROP_RELEASE_AFTER_SETTLE_126", True))
                            )

                            if release_enabled or force_drop_release_126:
                                # 124/125_ 핵심:
                                # 마지막 위치/회전은 그대로 두고, 흡착 FixedJoint만 제거해서
                                # 물리 적용된 상자가 중력으로 떨어지는지 확인한다.
                                if bool(DROP_RELEASE_AFTER_SETTLE_124):
                                    if bool(DROP_RELEASE_ZERO_VELOCITY_BEFORE_DETACH_124):
                                        zero_subtree_velocity(stage, task.box_move_path)
                                        print("  [DROP_RELEASE_124] zero velocity before detach = True")

                                    # 160_: release 직전 상자 yaw를 순간적으로 바꾸지 않는다.
                                    #       방향이 틀리면 틀린 상태로 로그에 남겨야 실제 로봇 동작 검증이 가능하다.
                                    if bool(globals().get("SLOT_MARKER_YAW_CHECK_LOG_160", True)):
                                        try:
                                            _slot_idx_for_yaw_160 = int(getattr(task, "_stack_slot_index", 0))
                                            # 170_: 진단 전용. release 직전 APalt_slot 정답지 대비 현재 box pose만 기록한다.
                                            #       어떤 transform/joint/RMPFlow/surface 상태도 수정하지 않는다.
                                            apalt_release_pose_diagnose_only_170(stage, task.box_move_path, _slot_idx_for_yaw_160, label="before_release_no_snap")
                                            axis_yaw_diagnostic_164(stage, task.box_move_path, _slot_idx_for_yaw_160, label="before_release_no_snap")
                                            check_box_yaw_against_slot_marker_160(stage, task.box_move_path, _slot_idx_for_yaw_160, label="before_release_no_snap")
                                        except Exception as _yaw_exc_160:
                                            print(f"[YAW_CHECK_160][WARN] before release failed: {type(_yaw_exc_160).__name__}: {_yaw_exc_160}")

                                    if bool(globals().get("RELEASE_DIAG_128_ENABLED", True)):
                                        print("========== [DIAG128_RELEASE_BEGIN] ==========")
                                        compact_box_physics_state_128(stage, task.box_move_path, "before_release")
                                        scan_stage_joints_128(stage, task.box_move_path, "before_release")

                                    release_ok = release_physics_attach_joint(stage, reason="drop_release_after_final_pose_128")

                                    if bool(globals().get("RELEASE_DIAG_128_ENABLED", True)):
                                        scan_stage_joints_128(stage, task.box_move_path, "after_joint_remove")
                                        compact_box_physics_state_128(stage, task.box_move_path, "after_joint_remove_before_dynamic_restore")
                                        if bool(globals().get("RELEASE_DIAG_FORCE_DYNAMIC_AFTER_DETACH_128", True)):
                                            force_box_dynamic_after_release_128(stage, task.box_move_path, verbose=True)
                                        compact_box_physics_state_128(stage, task.box_move_path, "after_dynamic_restore")
                                        compact_drop_observe_128(stage, task.box_move_path, "after_release_step0")

                                    attached = False
                                    released = True
                                    custom_carry_active = False
                                    custom_carry_phase = None
                                    attached_center_offset = None
                                    attach_center_z = None
                                    print(
                                        f"-surface off- DROP_RELEASE_128: FixedJoint removed={release_ok}. "
                                        "no snap/no teleport/no kinematic toggle. Box should fall by gravity if it is dynamic."
                                    )
                                    log_oribox_pose_tracker(stage, "after_drop_release_128_begin", force=True)

                                    if MULTI_ORIBOX_STACKING_ENABLED:
                                        _release_count_after_drop_164 = handle_stack_release_count_145(stage, task, label="drop_release_128")
                                        if _diag_stop_after_release_count_reached_164(_release_count_after_drop_164):
                                            print("[DIAG_STOP_AFTER_RELEASE_164] world.pause() after second release. Send this log for diagnosis.")
                                            task_done = True
                                            my_world.pause()
                                            was_playing = is_playing
                                            continue
                                    else:
                                        completed_box_roots.add(get_task_completed_root_path(task))
                                        check_and_trigger_forklift()

                                    # 바로 home으로 보내면 낙하 장면이 잘 안 보일 수 있으므로, 지정 step 동안 관찰한다.
                                    observe_steps_124 = int(max(0, DROP_RELEASE_OBSERVE_STEPS_124))
                                    log_interval_124 = int(max(1, DROP_RELEASE_LOG_INTERVAL_124))
                                    for drop_i_124 in range(observe_steps_124):
                                        try:
                                            update_vgc10_suction_anchor(robot)
                                        except Exception:
                                            pass
                                        my_world.step(render=True)
                                        if drop_i_124 % log_interval_124 == 0 or drop_i_124 == observe_steps_124 - 1:
                                            if bool(globals().get("RELEASE_DIAG_128_ENABLED", True)):
                                                compact_drop_observe_128(stage, task.box_move_path, f"drop_release_128:{drop_i_124}")
                                            else:
                                                log_oribox_pose_tracker(stage, f"drop_release_126:{drop_i_124}", force=True)

                                    # 134_: release 후 바로 종료 상태로 막지 않는다.
                                    # home-return을 사용할 때는 task_done=False 상태에서 start_home_return을 호출해야
                                    # 다음 프레임에서 returning_home 블록이 실행되어 로봇이 초기 자세로 돌아간다.
                                    if bool(DROP_RELEASE_RETURN_HOME_AFTER_OBSERVE_124) and bool(RETURN_HOME_AFTER_RELEASE):
                                        task_done = False
                                        print("[RESET_134_AFTER_RELEASE] drop observe done -> start home/initial return")
                                        start_home_return("drop_release_134_after_observe", success=True)
                                    else:
                                        task_done = True
                                    was_playing = is_playing
                                    continue

                                # fallback: 기존 safe release 방식. DROP_RELEASE_AFTER_SETTLE_124=False일 때만 사용.
                                attached = False
                                custom_carry_active = False
                                custom_carry_phase = None
                                print("-surface off- BoxAprop safe release: no physics toggle, no snap, keep current pose.")
                                log_oribox_pose_tracker(stage, "after_safe_release_before_stack_done", force=True)
                                if MULTI_ORIBOX_STACKING_ENABLED:
                                    _release_count_after_safe_164 = handle_stack_release_count_145(stage, task, label="safe_release")
                                    if _diag_stop_after_release_count_reached_164(_release_count_after_safe_164):
                                        print("[DIAG_STOP_AFTER_RELEASE_164] world.pause() after second safe release. Send this log for diagnosis.")
                                        task_done = True
                                        my_world.pause()
                                        was_playing = is_playing
                                        continue
                                else:
                                    completed_box_roots.add(get_task_completed_root_path(task))
                                    check_and_trigger_forklift()
                                task_done = False
                                print("[RESET_134_AFTER_SAFE_RELEASE] start home/initial return")
                                start_home_return("boxaprop_safe_release_no_physics_toggle_134", success=True)
                                was_playing = is_playing
                                continue

                            custom_carry_active = False
                            custom_carry_phase = None
                            task_done = True
                            if bool(VERTICAL_LIFT_PAUSE_AFTER_SUCCESS):
                                my_world.pause()
                            was_playing = is_playing
                            continue

                        # 50_: 순간이동 방지. slot 중심 snap은 기본 비활성화되어 있으며,
                        # 실제 로봇/흡착점이 이동한 현재 위치에서만 release한다.
                        snap_box_to_stack_slot_if_enabled(stage, task)
                        release_physics_attach_joint(stage, reason="cartesian_carry_release")
                        if BOX_DISABLE_PHYSICS_DURING_CARRY:
                            set_box_scripted_carry_mode(
                                stage,
                                task.box_move_path,
                                False,
                                reenable_physics=BOX_REENABLE_PHYSICS_AFTER_RELEASE,
                                verbose=True,
                            )
                        elif not PHYSICS_FIXED_JOINT_ATTACH_ENABLED:
                            if bool(BOX_KEEP_KINEMATIC_AFTER_RELEASE):
                                set_prim_kinematic(stage, task.box_path, True)
                                print("  [CENTER_TOP_LOCK_RELEASE] placed box kept kinematic=True after vertical-only test.")
                            else:
                                set_prim_kinematic(stage, task.box_path, False)
                        zero_subtree_velocity(stage, task.box_move_path)
                        if MULTI_ORIBOX_STACKING_ENABLED:
                            handle_stack_release_count_145(stage, task, label="cartesian_carry_release")
                        attached = False
                        released = True
                        custom_carry_active = False
                        custom_carry_phase = None
                        attached_center_offset = None
                        attach_center_z = None
                        print("-surface off- OriBoxA_ box release at current pose. no snap/no teleport.")

                        if RETURN_HOME_AFTER_RELEASE:
                            if RUN_CONTINUOUS_LOOP:
                                start_home_return("success_release_after_cartesian_carry", success=True)
                            else:
                                returning_home = True
                                home_start_joints = np.array(robot.get_joint_positions(), dtype=float)
                                home_target_joints = np.zeros_like(home_start_joints)
                                home_return_step = 0
                                print("[RETURN] box released. returning robot to initial joint pose...")

                was_playing = is_playing
                continue

            obs = task.get_observations()
            box_center = np.array(obs["oribox"]["position"], dtype=float)
            current_joints = obs["m0609_robot"]["joint_positions"]

            # controller에는 움직이는 박스 현재 위치가 아니라 초기 pick 위치를 계속 넣는다.
            # 그래야 흡착 후 박스가 따라 움직여도 로봇 목표가 흔들리지 않는다.
            actions = controller.forward(
                picking_position=task.pick_center,
                placing_position=task.goal_center,
                current_joint_positions=current_joints,
                end_effector_offset=EE_OFFSET,
            )
            robot.apply_action(actions)

            event = controller.get_current_event()
            suction_pos = update_vgc10_suction_anchor(robot)

            bbox = get_world_bbox_info(stage, task.box_path)
            close_gate, close_reason = should_attach_oribox(event, suction_pos, bbox, box_stopped=box_stopped_for_pick)

            # 10_ safety: 흡착이 안 된 상태로 VGC10이 박스 윗면을 계속 관통하면 바로 멈춘다.
            # 이 로그가 뜨면 흡착 판정이 너무 늦거나, tool0/VGC10 offset이 너무 낮은 것이다.
            if (
                PAUSE_IF_SUCTION_PENETRATES_WITHOUT_ATTACH
                and (not attached)
                and (not ever_attached)
                and bbox is not None
                and event in PICK_CLOSE_EVENTS
            ):
                z_gap_now = float(np.array(suction_pos, dtype=float)[2] - float(bbox["top_center"][2]))
                if z_gap_now < SUCTION_PENETRATION_STOP_Z_GAP:
                    print(
                        f"[STOP_PENETRATION] suction point is too far below box top before attach. "
                        f"z_gap={z_gap_now:.4f}, limit={SUCTION_PENETRATION_STOP_Z_GAP:.4f}, "
                        f"reason={close_reason}. returning home and retrying with refreshed bbox."
                    )
                    if RUN_CONTINUOUS_LOOP and LOOP_RETRY_AFTER_ATTACH_FAIL:
                        start_home_return("suction_penetration_before_attach", success=False)
                    else:
                        task_done = True
                        my_world.pause()
                    was_playing = is_playing
                    continue

            if (not attached) and event in (PICK_CLOSE_EVENTS | RETRY_CLOSE_EVENTS):
                try:
                    top_center = bbox["top_center"] if bbox is not None else box_center
                    now_dist = float(np.linalg.norm(np.array(suction_pos, dtype=float) - np.array(top_center, dtype=float)))
                    if now_dist < best_attach_dist:
                        best_attach_dist = now_dist
                        best_attach_reason = close_reason
                except Exception:
                    pass

            if (not attached) and (event in RETRY_CLOSE_EVENTS) and (not retry_logged):
                retry_logged = True
                print(f"-surface retry- no attach at event2/3, retrying near box best={best_attach_reason}")

            if not attached and close_gate:
                # 15_ 핵심: root 기준 offset이 아니라 bbox center 기준 offset을 저장한다.
                # 이렇게 해야 박스 루트가 바닥 기준이거나 이상한 xform op를 가져도
                # 실제 보이는 박스 중심이 suction point를 따라간다.
                bbox_for_attach = get_world_bbox_info(stage, task.box_path)
                attach_center = np.array(bbox_for_attach["center"], dtype=float) if bbox_for_attach is not None else np.array(box_center, dtype=float)
                grid_attach_center = _LAST_SUCTION_GRID_INFO.get("attach_center")
                if grid_attach_center is None:
                    grid_attach_center = np.array(suction_pos, dtype=float)
                grid_attach_center = np.array(grid_attach_center, dtype=float)
                attached_center_offset = attach_center - np.array(suction_pos, dtype=float)

                # 51_: 박스 물리를 끄거나 kinematic으로 바꾸지 않는다.
                # 9점 흡착 판정 평균점에 FixedJoint를 생성해 실제 물리 연결로 들어올린다.
                joint_ok = True
                if PHYSICS_FIXED_JOINT_ATTACH_ENABLED:
                    joint_ok = create_physics_attach_joint(stage, task.box_path, grid_attach_center)
                if not joint_ok:
                    print("[PHYSICS_ATTACH_ABORT] FixedJoint 생성 실패. 박스 물리를 끄는 fallback은 사용하지 않고 재시도합니다.")
                    if RUN_CONTINUOUS_LOOP and LOOP_RETRY_AFTER_ATTACH_FAIL:
                        start_home_return("physics_joint_attach_failed", success=False)
                    else:
                        task_done = True
                        my_world.pause()
                    was_playing = is_playing
                    continue
                zero_subtree_velocity(stage, task.box_move_path)

                attached = True
                ever_attached = True
                attach_center_z = float(attach_center[2])
                attach_steps = 0
                print(f"-surface on- OriBoxA_ box attach reason={close_reason}")
                print(f"             attached_center_offset={attached_center_offset}")
                try:
                    _slot_idx_attach_164 = int(getattr(task, "_stack_slot_index", stack_slot_index))
                    axis_yaw_diagnostic_164(stage, task.box_move_path, _slot_idx_attach_164, label="after_attach_fixed_joint_main")
                except Exception as _axis_attach_exc_164:
                    print(f"[BOX_AXIS_YAW_164][WARN] after_attach_main failed: {type(_axis_attach_exc_164).__name__}: {_axis_attach_exc_164}")

                if CUSTOM_CARRY_AFTER_ATTACH:
                    custom_carry_active = True
                    custom_phase_step = 0
                    custom_phase_index = 0
                    custom_min_center_z = float(attach_center_z)

                    if CUSTOM_CARRY_MODE == "JOINT_SWING":
                        custom_fixed_orientation = None
                        custom_targets = make_joint_swing_carry_targets(
                            stage,
                            robot,
                            task,
                            attach_center=attach_center,
                            attach_suction_pos=suction_pos,
                            attached_center_offset=attached_center_offset,
                        )
                        custom_carry_phase = custom_targets["phase_sequence"][0].get("name", "joint_lift")
                        print(
                            "[JOINT_CARRY_27_VERTICAL_FIRST_HALF_TURN] start. "
                            f"center_path={custom_targets.get('center_path')}, "
                            f"lift_delta={custom_targets['lift_delta']}, "
                            f"swing_delta={custom_targets['swing_delta']}, "
                            f"mirror_xy_estimate={custom_targets['mirror_xy_estimate']}, "
                            f"phases={[p['name'] for p in custom_targets.get('phase_sequence', [])]}"
                        )
                    else:
                        custom_carry_phase = "lift"
                        custom_fixed_orientation = get_current_ee_pose(robot)[1]
                        custom_targets = make_custom_carry_targets(
                            stage,
                            task,
                            attach_center=attach_center,
                            attach_suction_pos=suction_pos,
                            attached_center_offset=attached_center_offset,
                        )
                        if custom_targets.get("phase_sequence"):
                            custom_carry_phase = custom_targets["phase_sequence"][0].get("name", "lift")
                        cart_controller.reset()
                        print(
                            "[HYBRID_VERTICAL_JOINT1_REVERSE_117] start. vertical RMPFlow lift + joint_1 rotate + reverse vertical lower. "
                            f"safe_z={custom_targets['safe_z']:.3f}, "
                            f"lift_suction={custom_targets['lift_suction']}, "
                            f"move_suction={custom_targets['move_suction']}, "
                            f"lower_suction={custom_targets['lower_suction']}, "
                            f"place_center={custom_targets['place_center']}, "
                            f"phases={[p['name'] for p in custom_targets.get('phase_sequence', [])]}"
                        )

                    was_playing = is_playing
                    continue

                if RETURN_HOME_IMMEDIATELY_AFTER_ATTACH:
                    returning_home = True
                    home_start_joints = np.array(robot.get_joint_positions(), dtype=float)
                    home_target_joints = np.zeros_like(home_start_joints) if HOME_TARGET_JOINTS_CONFIG is None else np.array(HOME_TARGET_JOINTS_CONFIG, dtype=float)
                    home_return_step = 0
                    print("[RETURN] attach confirmed. returning-home-with-attached-box is disabled in 18_. continuing normal place/release.")
                    was_playing = is_playing
                    continue

            if (not attached) and (not ever_attached) and (not released) and event >= FAIL_IF_NOT_ATTACHED_EVENT:
                print(
                    f"[FAIL] suction attach failed before departure. "
                    f"best={best_attach_reason}, best_dist={best_attach_dist:.4f}. returning home and retrying with refreshed bbox."
                )
                if RUN_CONTINUOUS_LOOP and LOOP_RETRY_AFTER_ATTACH_FAIL:
                    start_home_return("attach_failed_before_departure", success=False)
                else:
                    task_done = True
                    my_world.pause()
                was_playing = is_playing
                continue

            if attached:
                attach_steps += 1
                bbox_for_release_check = get_world_bbox_info(stage, task.box_path)
                current_center_for_release = np.array(bbox_for_release_check["center"], dtype=float) if bbox_for_release_check is not None else np.array(box_center, dtype=float)
                lifted_enough = (attach_center_z is not None) and (float(current_center_for_release[2]) >= float(attach_center_z) + RELEASE_AFTER_LIFT_DELTA_Z)
                goal_xy_error = float(np.linalg.norm(current_center_for_release[:2] - np.array(task.goal_center, dtype=float)[:2]))
                goal_z_error = float(abs(float(current_center_for_release[2]) - float(task.goal_center[2])))
                near_goal_xy = goal_xy_error <= PLACE_RELEASE_XY_TOL
                near_goal_z = goal_z_error <= PLACE_RELEASE_Z_TOL
                # 22_ 핵심:
                # 21_에서는 near_goal_z까지 기다리면서 PickPlaceController가 하강(event=5/6)해
                # 박스를 계속 아래로 끌고 갔다. 이제 Z는 기다리지 않는다.
                # 충분히 들어 올린 뒤, 반대편 XY 좌표 근처에 도착하면 바로 release한다.
                should_release_now = (
                    attach_steps >= RELEASE_AFTER_ATTACH_MIN_STEPS
                    and event >= PLACE_RELEASE_EVENT
                    and lifted_enough
                    and near_goal_xy
                )

                if should_release_now:
                    if RELEASE_AT_CURRENT_POSE:
                        # 박스를 목표점으로 강제 이동하지 않는다. 현재 들린 위치에서 release한다.
                        pass
                    else:
                        move_box_center_to(
                            stage,
                            task.box_move_path,
                            desired_center=task.goal_center,
                            root_to_center_offset=task.box_root_to_center_offset,
                        )
                    release_physics_attach_joint(stage, reason="event_goal_release")
                    if BOX_DISABLE_PHYSICS_DURING_CARRY:
                        set_box_scripted_carry_mode(
                            stage,
                            task.box_move_path,
                            False,
                            reenable_physics=BOX_REENABLE_PHYSICS_AFTER_RELEASE,
                            verbose=True,
                        )
                    elif not PHYSICS_FIXED_JOINT_ATTACH_ENABLED:
                        set_prim_kinematic(stage, task.box_path, False)
                    zero_subtree_velocity(stage, task.box_move_path)

                    if MULTI_ORIBOX_STACKING_ENABLED:
                        handle_stack_release_count_145(stage, task, label="event_goal_release")

                    attached = False
                    released = True
                    attached_center_offset = None
                    attach_center_z = None
                    attach_steps = 0

                    print(f"-surface off- OriBoxA_ box release at stack goal event={event}, lifted_enough={lifted_enough}, goal_xy_error={goal_xy_error:.4f}, goal_z_error={goal_z_error:.4f}")

                    if RETURN_HOME_AFTER_RELEASE:
                        if RUN_CONTINUOUS_LOOP:
                            start_home_return("success_release_at_mirror_goal", success=True)
                        else:
                            returning_home = True
                            home_start_joints = np.array(robot.get_joint_positions(), dtype=float)
                            home_target_joints = np.zeros_like(home_start_joints)
                            home_return_step = 0
                            print("[RETURN] box released. returning robot to initial joint pose...")
                        was_playing = is_playing
                        continue

                else:
                    if event >= PLACE_RELEASE_EVENT and attach_steps % 10 == 0:
                        print(
                            f"  [wait_release_goal] event={event}, goal_xy_error={goal_xy_error:.4f}/{PLACE_RELEASE_XY_TOL:.4f}, "
                            f"goal_z_error={goal_z_error:.4f}/{PLACE_RELEASE_Z_TOL:.4f}, lifted={lifted_enough}"
                        )
                    if attached_center_offset is None:
                        bbox_for_follow = get_world_bbox_info(stage, task.box_path)
                        follow_center = np.array(bbox_for_follow["center"], dtype=float) if bbox_for_follow is not None else np.array(box_center, dtype=float)
                        attached_center_offset = follow_center - np.array(suction_pos, dtype=float)

                    desired_box_center = np.array(suction_pos, dtype=float) + np.array(attached_center_offset, dtype=float)

                    # place 하강 중에는 박스 중심이 목표 높이보다 과도하게 낮아지지 않게 막는다.
                    # event=6에서 release 전까지 계속 아래로 끌고 내려가면 관통처럼 보이므로 높이를 clamp한다.
                    if event >= PLACE_RELEASE_EVENT:
                        # 23_: 반대편 좌표로 가는 동안 박스가 아래로 끌려 내려가지 않도록
                        # 최소 높이를 attach 시점보다 6cm 이상 높은 위치와 goal 높이 중 큰 값으로 고정한다.
                        min_carry_z = float(task.goal_center[2])
                        if attach_center_z is not None:
                            min_carry_z = max(min_carry_z, float(attach_center_z) + RELEASE_AFTER_LIFT_DELTA_Z)
                        desired_box_center[2] = max(float(desired_box_center[2]), min_carry_z)

                    if BOX_DISABLE_PHYSICS_DURING_CARRY:
                        # 매 프레임 물리 엔진이 다시 속도를 주지 못하게 carry mode와 속도를 유지한다.
                        set_box_scripted_carry_mode(stage, task.box_move_path, True, verbose=False)

                    if PHYSICS_FIXED_JOINT_ATTACH_ENABLED:
                        # 51_: 물리 연결 중에는 박스 transform을 직접 이동하지 않는다.
                        ok_follow_set = True
                    else:
                        ok_follow_set = move_box_center_to(
                            stage,
                            task.box_move_path,
                            desired_center=desired_box_center,
                            root_to_center_offset=task.box_root_to_center_offset,
                        )
                        if not BOX_DISABLE_PHYSICS_DURING_CARRY:
                            set_prim_kinematic(stage, task.box_path, True)
                        zero_subtree_velocity(stage, task.box_move_path)

                    if DEBUG_MOVE_PARENT_AND_FOLLOW:
                        bbox_after_follow = get_world_bbox_info(stage, task.box_path)
                        if bbox_after_follow is not None:
                            actual_center = np.array(bbox_after_follow["center"], dtype=float)
                            desired_center = np.array(desired_box_center, dtype=float)
                            follow_error = float(np.linalg.norm(actual_center - desired_center))
                            if follow_error > FOLLOW_ERROR_WARN_TOL:
                                print(
                                    f"[FOLLOW_WARN] move_path={task.box_move_path} ok_set={ok_follow_set} "
                                    f"desired_center=({desired_center[0]:.3f},{desired_center[1]:.3f},{desired_center[2]:.4f}) "
                                    f"actual_center=({actual_center[0]:.3f},{actual_center[1]:.3f},{actual_center[2]:.4f}) "
                                    f"follow_error={follow_error:.4f}>{FOLLOW_ERROR_WARN_TOL:.4f}"
                                )

            bbox_now = get_world_bbox_info(stage, task.box_path)
            box_center_now = bbox_now["center"] if bbox_now is not None else box_center
            top_center_now = bbox_now["top_center"] if bbox_now is not None else box_center_now

            if not np.all(np.isfinite(box_center_now)) or abs(float(box_center_now[2])) > 10.0:
                print(f"[STOP] box pose abnormal: {box_center_now}. pausing world.")
                task_done = True
                my_world.pause()

            if controller.is_done():
                if RUN_CONTINUOUS_LOOP and (not released) and (not custom_carry_active):
                    print("[LOOP_CONTROLLER_DONE] PickPlaceController ended without successful custom carry. returning home and retrying.")
                    start_home_return("controller_done_without_success", success=False)
                    was_playing = is_playing
                    continue
                else:
                    print("[완료] Pick & Place sequence done")
                    task_done = True
                    my_world.pause()

            dist_top = float(np.linalg.norm(np.array(top_center_now) - np.array(suction_pos)))
            print(
                f"  [event={event}] "
                f"box_center=({box_center_now[0]:.3f},{box_center_now[1]:.3f},{box_center_now[2]:.4f}) "
                f"box_top=({top_center_now[0]:.3f},{top_center_now[1]:.3f},{top_center_now[2]:.4f}) "
                f"suction=({suction_pos[0]:.3f},{suction_pos[1]:.3f},{suction_pos[2]:.4f}) "
                f"dist_top={dist_top:.4f} gate={close_gate} reason={close_reason} "
                f"box_stopped={box_stopped_for_pick} pick_started={pick_started} "
                f"surface={'ON' if attached else 'OFF'} "
                f"gripped={[task.box_move_path] if attached else []}"
            )

        was_playing = is_playing

    simulation_app.close()


if __name__ == "__main__":
    main()

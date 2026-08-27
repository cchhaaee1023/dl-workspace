import cv2
import mediapipe as mp
import time

# ==============================================================================
# 1. MediaPipe Tasks API 핵심 클래스 로드
# ==============================================================================
# BaseOptions: 모델 파일 경로(.task) 및 디바이스(CPU/GPU) 설정 담당
BaseOptions = mp.tasks.BaseOptions

# HandLandmarker: 실제 손 관절 위치를 추론하는 메인 클래스
HandLandmarker = mp.tasks.vision.HandLandmarker

# HandLandmarkerOptions: 신뢰도 임계값, 인식 모드, 최대 손 개수 등을 세팅하는 옵션 클래스
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

# VisionRunningMode: 입력 데이터 형태 정의 (IMAGE: 정지영상, VIDEO: 비디오 파일, LIVE_STREAM: 실시간 웹캠)
VisionRunningMode = mp.tasks.vision.RunningMode


# ==============================================================================
# 2. 손 관절(Landmark) 연결선 정의 (시각화용)
# ==============================================================================
# 21개 관절 점 인덱스를 기반으로 손가락 마디마디를 잇는 튜플 리스트
HAND_CONNECTIONS = [
    # 엄지손가락 (Wrist 0 -> 손목 base 1 -> 관절 2, 3 -> 손끝 4)
    (0, 1), (1, 2), (2, 3), (3, 4),

    # 검지손가락 (Wrist 0 -> 검지 뿌리 5 -> 관절 6, 7 -> 손끝 8)
    (0, 5), (5, 6), (6, 7), (7, 8),

    # 중지손가락 (검지 뿌리 5 -> 중지 뿌리 9 -> 관절 10, 11 -> 손끝 12)
    (5, 9), (9, 10), (10, 11), (11, 12),

    # 약지손가락 (중지 뿌리 9 -> 약지 뿌리 13 -> 관절 14, 15 -> 손끝 16)
    (9, 13), (13, 14), (14, 15), (15, 16),

    # 새끼손가락 (약지 뿌리 13 -> 새끼 뿌리 17 -> 관절 18, 19 -> 손끝 20 + 손목 0 연결)
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]


# ==============================================================================
# 3. HandLandmarker 옵션 설정 및 모델 로드
# ==============================================================================
options = HandLandmarkerOptions(
    # 다운로드받은 모델 파일(.task)의 상대/절대 경로 지정
    base_options=BaseOptions(
        model_asset_path='hand_landmarker.task',
        delegate=BaseOptions.Delegate.CPU,
    ),

    # 비디오 스트림 처리 모드로 설정 (동적 타임스탬프와 함께 연속 프레임 추적)
    running_mode=VisionRunningMode.VIDEO,

    # 화면 내 감지할 최대 손 개수 설정 (1로 설정 시 가장 명확한 손 1개만 추적)
    num_hands=1,

    # 손 탐지(Detection) 최소 신뢰도 (0.5 = 50% 이상 확신할 때 손으로 인식)
    min_hand_detection_confidence=0.5,

    # 손 추적(Tracking) 최소 신뢰도 (프레임 간 손의 위치를 계속 추적하는 임계값)
    min_tracking_confidence=0.5
)

# 기본 연결된 웹캠 장치 열기 (0: 기본 카메라)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다 - 권한(개인정보 보호 및 보안 > 카메라)을 확인하세요")


# ==============================================================================
# 4. 실시간 추론 및 시각화 루프
# ==============================================================================
# try/finally로 감싸서 루프 중 예외가 나도 카메라와 창은 반드시 해제되도록 함
try:
    # 'with' 문을 통해 Landmarker 객체를 안전하게 생성 (작업 종료 시 메모리 자원 자동 해제)
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            # 웹캠으로부터 1개 프레임 읽어오기 (ret: 성공여부 Boolean, frame: BGR 이미지 배열)
            ret, frame = cap.read()
            if not ret:
                print("카메라 프레임을 읽어올 수 없습니다.")
                break

            # 거울 모드 처리를 위한 좌우 반전 (1: 좌우 반전)
            frame = cv2.flip(frame, 1)

            # OpenCV의 BGR 색상 채널을 MediaPipe가 지원하는 RGB 채널로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # NumPy 배열 이미지를 MediaPipe의 전용 Image 객체로 래핑
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Tasks API VIDEO 모드 필수값: 밀리초(ms) 단위의 단조 증가 타임스탬프 생성
            frame_timestamp_ms = int(time.time() * 1000)

            # RGB 이미지와 타임스탬프를 모델에 전달하여 3D 관절 추론 실행
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # 원본 프레임의 높이(h), 너비(w) 추출 (정규화 좌표 -> 픽셀 좌표 변환용)
            h, w, _ = frame.shape

            # 손이 화면에서 성공적으로 감지되었는지 확인
            if result.hand_landmarks:
                # 감지된 손 개수만큼 반복 (num_hands=1 설정으로 최대 1회 실행)
                for hand_landmarks in result.hand_landmarks:
                    pixel_coords = []  # 21개 관절의 (x, y) 픽셀 좌표를 담을 리스트

                    # --------------------------------------------------------------
                    # 4-1. 관절 점(Landmark) 위치 계산 및 출력
                    # --------------------------------------------------------------
                    for idx, lm in enumerate(hand_landmarks):
                        # lm.x, lm.y는 0.0~1.0 사이로 정규화된 값이므로 해상도(w, h)를 곱해 픽셀 단위로 변환
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        pixel_coords.append((cx, cy))

                        if idx == 4:
                            # 4번 : 엄지 손가락 끝 (빨간색)
                            cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)
                        elif idx == 8:
                            # 8번 : 검지 손가락 끝 (초록색)
                            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
                        else:
                            # 나머지 관절 위치에 녹색 원 그리기
                            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                    # --------------------------------------------------------------
                    # 4-2. 관절 마디 연결선 그리기
                    # --------------------------------------------------------------
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        # HAND_CONNECTIONS 상의 두 관절 픽셀 좌표 획득
                        p1 = pixel_coords[start_idx]
                        p2 = pixel_coords[end_idx]

                        # 두 점 사이에 파란색 직선 그리기 (색상: BGR (255,0,0) / 두께: 2px)
                        cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # 처리 결과 프레임을 "Hand Recognition" 윈도우 창에 표시
            cv2.imshow("Hand Recognition", frame)

            # 5밀리초 동안 키 입력을 대기하고 8비트 마스크 적용
            key = cv2.waitKey(5) & 0xFF
            # ESC 키(ASCII 27) 또는 'q' 키를 누르면 루프 종료
            if key == 27 or key == ord('q'):
                break

            # 창의 X 버튼으로 닫은 경우도 루프 종료 (WND_PROP_VISIBLE이 0이 됨)
            if cv2.getWindowProperty("Hand Recognition", cv2.WND_PROP_VISIBLE) < 1:
                break
finally:
    # ==========================================================================
    # 5. 자원 해제 (정상 종료든 예외든 항상 실행됨)
    # ==========================================================================
    # 웹캠 장치 점유 해제
    cap.release()

    # 오픈되어 있는 모든 OpenCV 그래픽 창 닫기
    cv2.destroyAllWindows()

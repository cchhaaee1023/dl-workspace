# OpenCV 라이브러리: 비디오 프레임 캡처, 좌표 변환, 화면 시각화 담당
import cv2

# MediaPipe 라이브러리: 손 관절(21개 랜드마크) 추론을 담당
import mediapipe as mp

# PyAutoGUI 라이브러리: 파이썬으로 진짜 마우스 이동 및 클릭 제어
import pyautogui

# time 라이브러리: MediaPipe Tasks API에 넘겨줄 타임스탬프 생성용
import time

# math 라이브러리: 엄지와 검지 손가락 끝 사이의 유클리드 거리를 계산용
import math

# ==============================================================================
# 1. PyAutoGUI 환경 설정 및 모니터 해상도 구하기
# ==============================================================================
# 마우스 커서가 화면 모서리에 닿았을 때 발생할 수 있는 파이썬 예외 강제 종료 방지
pyautogui.FAILSAFE = False

# 마우스 동작 간 기본 대기 시간을 0.01초로 짧게 설정하여 커서 반응 속도 향상
pyautogui.PAUSE = 0.01

# 현재 사용 중인 모니터 화면의 전체 가로(screen_w), 세로(screen_h) 해상도 획득
screen_w, screen_h = pyautogui.size()

# ==============================================================================
# 2. MediaPipe Tasks API 설정 및 모델 로드
# ==============================================================================
# BaseOptions: .task 모델 파일 경로 지정용 클래스
BaseOptions = mp.tasks.BaseOptions

# HandLandmarker: 실제 손 관절 좌표 추론 클래스
HandLandmarker = mp.tasks.vision.HandLandmarker

# HandLandmarkerOptions: 신뢰도, 추론 모드, 손 개수 등 세부 옵션 지정 클래스
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

# VisionRunningMode: 비디오 프레임 스트림 방식 선택용 클래스
VisionRunningMode = mp.tasks.vision.RunningMode

# 손 관절 마디 연결 정보 정의 (시각화 목적)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 엄지손가락
    (0, 5), (5, 6), (6, 7), (7, 8),        # 검지손가락
    (5, 9), (9, 10), (10, 11), (11, 12),   # 중지손가락
    (9, 13), (13, 14), (14, 15), (15, 16), # 약지손가락
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # 새끼손가락
]

# HandLandmarker 모델 세부 옵션 지정
options = HandLandmarkerOptions(
    # 로컬 경로에 있는 모델(.task) 지정
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    # 실시간 비디오 스트림 처리 모드로 설정
    running_mode=VisionRunningMode.VIDEO,
    # 최대 인식할 손 개수 (마우스 제어이므로 1개로 제한)
    num_hands=1,
    # 최초 손 감지 신뢰도 임계값
    min_hand_detection_confidence=0.5,
    # 지속 손 추적 신뢰도 임계값
    min_tracking_confidence=0.5
)

# 기본 웹캠 캡처 객체 생성 (0번 카메라)
cap = cv2.VideoCapture(0)

# 이전 프레임의 마우스 커서 위치 기록용 변수 (부드러운 보정 계산용)
prev_x, prev_y = 0, 0

# 마우스 커서의 떨림 현상을 완화하기 위한 보정 비율 (값이 클수록 부드럽지만 약간의 딜레이 발생)
smoothing = 5

# 클릭 상태를 유지/해제하여 프레임마다 연타로 클릭되는 현상을 방지하는 플래그
is_clicked = False

# 카메라 가장자리 여백 영역 설정 (손을 웹캠 끝까지 움직이지 않아도 모니터 구석에 닿도록 함)
margin = 100 

# ==============================================================================
# 3. 실시간 가상 마우스 루프 시작
# ==============================================================================
# Landmarker 인스턴스 자동 자원 관리를 위한 with 블록 시작
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        # 웹캠 프레임 가져오기 (ret: 성공 여부, frame: BGR 이미지)
        ret, frame = cap.read()
        if not ret:
            break

        # 사용자가 거울 보듯 자연스럽게 조작하도록 좌우 반전
        frame = cv2.flip(frame, 1)
        
        # 현재 카메라 프레임의 높이(h), 너비(w) 추출
        h, w, _ = frame.shape
        
        # OpenCV의 BGR 이미지를 MediaPipe용 RGB 이미지로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # NumPy 이미지 배열을 MediaPipe 전용 Image 객체로 래핑
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # VIDEO 모드 필수 요구사항: 밀리초(ms) 단위의 단조 증가 타임스탬프 계산
        frame_timestamp_ms = int(time.time() * 1000)
        
        # MediaPipe 모델로 손 좌표 추론 실행
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # 카메라 화면 상에 노란색 가상 조작 영역 Box 표시
        cv2.rectangle(frame, (margin, margin), (w - margin, h - margin), (255, 255, 0), 2)

        # 손 관절 추론 결과가 존재하는 경우
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # 21개 관절의 픽셀 좌표를 보관할 리스트
                pixel_coords = []
                
                # 랜드마크 정규화 좌표(0.0~1.0)를 해상도에 맞는 픽셀 좌표로 환산하여 시각화
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pixel_coords.append((cx, cy))
                    # 관절 지점에 초록색 점 그리기
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                # 손가락 관절 간 파란색 선으로 연결선 그리기
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, pixel_coords[start_idx], pixel_coords[end_idx], (255, 0, 0), 2)

                # 마우스 커서 조작 및 클릭 판정에 사용할 엄지 끝(4번)과 검지 끝(8번) 좌표 추출
                thumb_x, thumb_y = pixel_coords[4]
                index_x, index_y = pixel_coords[8]

                # --------------------------------------------------------------
                # 3-1. 마우스 이동 처리 (검지 손가락 끝 좌표 기준)
                # --------------------------------------------------------------
                # 카메라 조작 영역(margin 적용 영역) 좌표를 실제 모니터 해상도 좌표 비율로 선형 매핑
                target_x = (index_x - margin) / (w - 2 * margin) * screen_w
                target_y = (index_y - margin) / (h - 2 * margin) * screen_h

                # 모니터 화면 바깥으로 커서가 벗어나지 않도록 좌표값 상하한 제한
                target_x = max(0, min(screen_w, target_x))
                target_y = max(0, min(screen_h, target_y))

                # 이동 보정 알고리즘: 이전 좌표와 목표 좌표 사이를 보정하여 손떨림으로 인한 마우스 튐 방지
                curr_x = prev_x + (target_x - prev_x) / smoothing
                curr_y = prev_y + (target_y - prev_y) / smoothing

                # PyAutoGUI를 통해 실제 OS 마우스 위치 이동
                pyautogui.moveTo(curr_x, curr_y)
                
                # 다음 프레임 보정을 위한 현재 위치 저장
                prev_x, prev_y = curr_x, curr_y

                # --------------------------------------------------------------
                # 3-2. 마우스 좌클릭 처리 (검지와 엄지 손가락 끝 간 거리 계산)
                # --------------------------------------------------------------
                # 2차원 피타고라스 정리(유클리드 거리)로 두 손가락 끝의 픽셀 거리 계산
                distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
                
                # 엄지와 검지 사이를 연결하는 노란색 직선 그리기 (집게 동작 시각화)
                cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 255), 2)

                # 거리가 30픽셀 미만으로 좁혀지면(손가락 맞닿음/집게 동작) 클릭 인식
                if distance < 30:
                    # 클릭 동작 인식 시 검지 끝에 빨간색 동그라미 표시
                    cv2.circle(frame, (index_x, index_y), 10, (0, 0, 255), -1)
                    
                    # 단발성 클릭 처리를 위해 is_clicked 플래그 사용
                    if not is_clicked:
                        pyautogui.click() # 좌클릭 실행
                        is_clicked = True  # 클릭 중 상태로 전환
                else:
                    # 손가락을 떼면 클릭 상태 해제
                    is_clicked = False

        # 가상 마우스 조작 화면 출력
        cv2.imshow("Virtual Mouse", frame)

        # 5밀리초 동안 키 입력 대기 ('q' 키 또는 ESC 입력 시 반복 탈출)
        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):
            break

# ==============================================================================
# 4. 자원 해제
# ==============================================================================
# 웹캠 장치 연결 해제
cap.release()

# 모든 OpenCV 그래픽 창 닫기
cv2.destroyAllWindows()
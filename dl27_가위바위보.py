# OpenCV 라이브러리: 비디오 프레임 처리 및 화면 출력을 담당
import cv2

# MediaPipe 라이브러리: 손 관절(Landmark) 추론 및 컴퓨터 비전 기능 제공
import mediapipe as mp

# random 라이브러리: 컴퓨터의 가위/바위/보 무작위 선택에 사용
import random

# time 라이브러리: MediaPipe Tasks API 입력에 필요한 밀리초 타임스탬프 계산
import time

# ==============================================================================
# 1. MediaPipe Tasks API 필수 모듈 및 설정 로드
# ==============================================================================
# BaseOptions: 모델 파일 경로(.task) 및 CPU/GPU 디바이스 지정 옵션
BaseOptions = mp.tasks.BaseOptions

# HandLandmarker: 21개 손 관절 좌표를 추론하는 메인 클래스
HandLandmarker = mp.tasks.vision.HandLandmarker

# HandLandmarkerOptions: 추론 신뢰도, 감지할 손 개수, 실행 모드를 세팅하는 클래스
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

# VisionRunningMode: 입력 형태 설정 (IMAGE: 이미지, VIDEO: 비디오 파일, LIVE_STREAM: 비동기 스트림)
VisionRunningMode = mp.tasks.vision.RunningMode

# 손 관절 뼈대를 이어줄 21개 랜드마크 인덱스 쌍 정의 (시각화 목적)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 엄지손가락 마디 연결선
    (0, 5), (5, 6), (6, 7), (7, 8),        # 검지손가락 마디 연결선
    (5, 9), (9, 10), (10, 11), (11, 12),   # 중지손가락 마디 연결선
    (9, 13), (13, 14), (14, 15), (15, 16), # 약지손가락 마디 연결선
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # 새끼손가락 마디 연결선
]

# ==============================================================================
# 2. 손지형(가위, 바위, 보) 판별 함수
# ==============================================================================
# 21개 관절 좌표(landmarks)를 전달받아 가위/바위/보 또는 알 수 없음 문자열을 반환하는 함수
def classify_rps(landmarks):
    # 검지 뿌리(5)와 새끼 뿌리(17)의 x좌표 위치를 비교하여 손바닥이 바라보는 방향 판별
    # 검지 뿌리가 오른쪽에 있으면 오른손, 왼쪽에 있으면 왼손 기준 적용
    if landmarks[5].x > landmarks[17].x:
        # 오른손 기준: 엄지 끝(4)의 x좌표가 엄지 마디(3)보다 오른쪽(더 큼)에 있으면 펴진 상태
        thumb_open = landmarks[4].x > landmarks[3].x
    else:
        # 왼손 기준: 엄지 끝(4)의 x좌표가 엄지 마디(3)보다 왼쪽(더 작음)에 있으면 펴진 상태
        thumb_open = landmarks[4].x < landmarks[3].x

    # 나머지 4개 손가락은 y축 좌표 비교 (화면 상단이 0, 하단이 1이므로 y값이 작을수록 위로 펴짐)
    # 검지 손가락 끝(8)이 중간 마디(6)보다 위(y값이 작음)에 있는지 확인
    index_open = landmarks[8].y < landmarks[6].y
    
    # 중지 손가락 끝(12)이 중간 마디(10)보다 위에 있는지 확인
    middle_open = landmarks[12].y < landmarks[10].y
    
    # 약지 손가락 끝(16)이 중간 마디(14)보다 위에 있는지 확인
    ring_open = landmarks[16].y < landmarks[14].y
    
    # 새끼 손가락 끝(20)이 중간 마디(18)보다 위에 있는지 확인
    pinky_open = landmarks[20].y < landmarks[18].y

    # 가위 조건 1: (엄지+검지)가 펴지고 중지, 약지, 새끼는 굽혀진 경우 (권총 모양)
    scissors_thumb_index = thumb_open and index_open and (not middle_open) and (not ring_open) and (not pinky_open)
    
    # 가위 조건 2: (검지+중지)가 펴지고 엄지, 약지, 새끼는 굽혀진 경우 (V 자 모양)
    scissors_index_middle = (not thumb_open) and index_open and middle_open and (not ring_open) and (not pinky_open)

    # 가위 조건 3: (약지 + 소지)가 펴지고 엄지, 검지, 중지는 굽혀진 경우 (작은 v 자 모양)
    scissors_index_middle2 = (not thumb_open) and (not index_open) and (not middle_open) and ring_open and pinky_open
    
    # 가위 조건 1 또는 2 중 하나라도 만족 시 "가위" 반환
    if scissors_thumb_index or scissors_index_middle or scissors_index_middle2:
        return "가위"
    
    # 보 조건: 엄지를 포함한 5개 손가락이 모두 펴진 경우 "보" 반환
    elif thumb_open and index_open and middle_open and ring_open and pinky_open:
        return "보"
    
    # 바위 조건: 엄지를 포함한 5개 손가락이 모두 굽혀진 경우 "바위" 반환
    elif (not thumb_open) and (not index_open) and (not middle_open) and (not ring_open) and (not pinky_open):
        return "바위"
    
    # 위 조건들에 해당하지 않는 부정확한 손짓일 경우 "알 수 없음" 반환
    else:
        return "알 수 없음"

# ==============================================================================
# 3. 메인 가위바위보 실행 및 UI 설정
# ==============================================================================
# MediaPipe HandLandmarker 모델 파라미터 및 인스턴스 옵션 구성
options = HandLandmarkerOptions(
    # 로컬 경로에 저장된 손 인식 모델 파일(.task) 지정
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    # 웹캠 비디오 프레임 스트림 처리를 위한 VIDEO 모드 선택
    running_mode=VisionRunningMode.VIDEO,
    # 추론할 최대 손 개수 (1개로 제한하여 오작동 방지)
    num_hands=1,
    # 최초 손 탐지 신뢰도 임계값 (50% 이상)
    min_hand_detection_confidence=0.5,
    # 프레임 간 손 추적 신뢰도 임계값 (50% 이상)
    min_tracking_confidence=0.5
)

# 0번 인덱스의 기본 웹캠 디바이스 연결
cap = cv2.VideoCapture(0)

# 가위바위보 게임에서 사용할 무작위 선택 리스트
choices = ["가위", "바위", "보"]
# choices = ["Sissors", "Rock", "Paper"]  # 영어로 변경 가능

# 컴퓨터가 선택한 낸 값 저장 변수 초기화
computer_choice = None

# 플레이어가 선택한 낸 값 저장 변수 초기화
user_choice = None

# 화면에 표시될 게임 결과 텍스트 초기화
game_result = "Space키를 눌러 가위바위보 시작!"

# 사용자 누적 점수 변수 초기화
score_user = 0

# 컴퓨터 누적 점수 변수 초기화
score_computer = 0

# HandLandmarker 인스턴스를 안전하게 생성하고 자원을 관리하는 with 블록 시작
with HandLandmarker.create_from_options(options) as landmarker:
    # 카메라 장치가 정상적으로 열려 있는 동안 지속 반복
    while cap.isOpened():
        # 카메라로부터 프레임 1개를 읽어옴 (ret: 성공여부, frame: BGR 이미지)
        ret, frame = cap.read()
        
        # 프레임을 읽어오지 못했으면(카메라 연결 끊김 등) 반복문 탈출
        if not ret:
            break

        # 사용자가 거울처럼 보게 하기 위해 좌우 반전 처리
        frame = cv2.flip(frame, 1)
        
        # 화면에 관절을 그리기 위해 프레임의 높이(h)와 너비(w) 추출
        h, w, _ = frame.shape
        
        # OpenCV의 BGR 색상 채널을 MediaPipe용 RGB 색상 채널로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # NumPy 이미지 배열을 MediaPipe 전용 Image 객체로 전환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # VIDEO 모드에 필수적으로 필요한 밀리초(ms) 단위의 타임스탬프 계산
        frame_timestamp_ms = int(time.time() * 1000)
        
        # MediaPipe 모델에 프레임과 타임스탬프를 전달하여 손 관절 추론 수행
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # 손 감지가 되지 않았을 때의 기본 상태 텍스트
        current_hand_gesture = "손을 올려주세요"

        # 손 관절 추론 결과가 존재할 경우
        if result.hand_landmarks:
            # 감지된 손 개수만큼 반복 (num_hands=1이므로 1회 실행)
            for hand_landmarks in result.hand_landmarks:
                # 21개 랜드마크의 화면 픽셀 좌표를 담을 리스트
                pixel_coords = []
                
                # 랜드마크 21개 좌표를 순회하며 픽셀 위치 계산
                for lm in hand_landmarks:
                    # 정규화된 0.0~1.0 좌표를 실제 화면 해상도 픽셀 좌표로 변환
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pixel_coords.append((cx, cy))
                    
                    # 관절 위치에 초록색 원 그리기
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                # 연결선 정의 배열(HAND_CONNECTIONS)에 따라 파란색 뼈대 그리기
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, pixel_coords[start_idx], pixel_coords[end_idx], (255, 0, 0), 2)

                # 현재 캡처된 손 좌표를 분류 함수로 전달해 가위/바위/보 판단
                current_hand_gesture = classify_rps(hand_landmarks)

        # ======================================================================
        # 4. 게임 화면 UI 및 오버레이 텍스트 출력
        # ======================================================================
        # 상단 누적 스코어판 출력 (흰색)
        cv2.putText(frame, f"Score - You: {score_user} | Com: {score_computer}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 현재 카메라에 인식 중인 손 모양 텍스트 출력 (노란색)
        cv2.putText(frame, f"Detected Hand: {current_hand_gesture}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 승패 및 안내 문구 출력 (초록색)
        cv2.putText(frame, f"Result: {game_result}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 컴퓨터의 선택이 존재할 경우 컴퓨터 선택값 출력 (연보라색)
        if computer_choice:
            cv2.putText(frame, f"Com Choice: {computer_choice}", (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 250), 2)

        # 하단 조작법 안내 문구 출력 (회색)
        cv2.putText(frame, "Press 'Space' to Play / 'q' or ESC to Exit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # 완성된 이미지 프레임을 창에 표시
        cv2.imshow("Rock Paper Scissors Game", frame)

        # 5밀리초 동안 키보드 입력 대기
        key = cv2.waitKey(5) & 0xFF
        
        # 스페이스바(ASCII 32) 입력 시 가위바위보 승패 판정 실행
        if key == 32:
            # 인식된 손 모양이 가위, 바위, 보 중 하나일 때만 게임 진행
            if current_hand_gesture in choices:
                # 사용자의 현재 손 모양 저장
                user_choice = current_hand_gesture
                
                # 컴퓨터의 선택 무작위 추출
                computer_choice = random.choice(choices)

                # 사용자 선택과 컴퓨터 선택이 같은 경우 (무승부)
                if user_choice == computer_choice:
                    game_result = f"Draw! (You: {user_choice} vs Com: {computer_choice})"
                
                # 사용자가 이기는 경우 (가위>보, 바위>가위, 보>바위)
                elif (user_choice == "가위" and computer_choice == "보") or \
                    (user_choice == "바위" and computer_choice == "가위") or \
                    (user_choice == "보" and computer_choice == "바위"):
                    game_result = f"You Win! (You: {user_choice} vs Com: {computer_choice})"
                    score_user += 1 # 사용자 점수 1점 추가
                
                # 컴퓨터가 이기는 경우
                else:
                    game_result = f"You Lose! (You: {user_choice} vs Com: {computer_choice})"
                    score_computer += 1 # 컴퓨터 점수 1점 추가
            
            # 인식 상태가 "알 수 없음"이거나 손을 올리지 않은 경우
            else:
                game_result = "Hand position unclear! Try again."

        # 'q' 키 또는 ESC(ASCII 27) 키 입력 시 게임 루프 종료
        elif key == 27 or key == ord('q'):
            break

# ==============================================================================
# 5. 메인 루프 종료 후 자원 해제
# ==============================================================================
# OpenCv 카메라 비디오 스트림 연결 해제
cap.release()

# 모든 OpenCV 그래픽 창 닫기
cv2.destroyAllWindows()
# OpenCV 라이브러리: 비디오 프레임 처리 및 화면 출력을 담당
import cv2

# MediaPipe 라이브러리: 손 관절(Landmark) 추론 및 컴퓨터 비전 기능 제공
import mediapipe as mp

# time 라이브러리: MediaPipe Tasks API 입력에 필요한 밀리초 타임스탬프 계산
import time

# ==============================================================================
# 1. MediaPipe Tasks API 필수 모듈 및 설정 로드
# ==============================================================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 엄지손가락 마디 연결선
    (0, 5), (5, 6), (6, 7), (7, 8),        # 검지손가락 마디 연결선
    (5, 9), (9, 10), (10, 11), (11, 12),   # 중지손가락 마디 연결선
    (9, 13), (13, 14), (14, 15), (15, 16), # 약지손가락 마디 연결선
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # 새끼손가락 마디 연결선
]

# ==============================================================================
# 2. 손지형(가위, 바위, 보) 판별 함수 (OpenCV 깨짐 방지를 위해 영문 반환)
# ==============================================================================
def classify_rps(landmarks):
    # 검지 뿌리(5)와 새끼 뿌리(17)의 x좌표 위치를 비교하여 손바닥 방향 판별
    if landmarks[5].x > landmarks[17].x:
        thumb_open = landmarks[4].x > landmarks[3].x
    else:
        thumb_open = landmarks[4].x < landmarks[3].x

    index_open = landmarks[8].y < landmarks[6].y
    middle_open = landmarks[12].y < landmarks[10].y
    ring_open = landmarks[16].y < landmarks[14].y
    pinky_open = landmarks[20].y < landmarks[18].y

    # 가위 조건
    scissors_thumb_index = thumb_open and index_open and (not middle_open) and (not ring_open) and (not pinky_open)
    scissors_index_middle = (not thumb_open) and index_open and middle_open and (not ring_open) and (not pinky_open)
    scissors_index_middle2 = (not thumb_open) and (not index_open) and (not middle_open) and ring_open and pinky_open

    if scissors_thumb_index or scissors_index_middle or scissors_index_middle2:
        return "Scissors"
    elif thumb_open and index_open and middle_open and ring_open and pinky_open:
        return "Paper"
    elif (not thumb_open) and (not index_open) and (not middle_open) and (not ring_open) and (not pinky_open):
        return "Rock"
    else:
        return "Unknown"

# ==============================================================================
# 3. 메인 가위바위보 실행 및 UI 설정
# ==============================================================================
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,  # 두 손을 동시에 감지하도록 2로 변경
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

choices = ["가위", "바위", "보"]

# 점수 기록 변수 (왼손 vs 오른손)
score_left = 0
score_right = 0
game_result = "Show 2 hands & Press SPACE!"

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(time.time() * 1000)
        
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # 감지된 각 손의 상태 저장 dictionary
        detected_gestures = {"Left": "None", "Right": "None"}

        if result.hand_landmarks and result.handedness:
            # 인식된 손 랜드마크와 왼손/오른손 정보를 함께 순회
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                # MediaPipe의 왼손/오른손 라벨 확인
                hand_label = result.handedness[idx][0].category_name  # "Left" 또는 "Right"
                
                pixel_coords = []
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pixel_coords.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, pixel_coords[start_idx], pixel_coords[end_idx], (255, 0, 0), 2)

                # 해당 손의 가위/바위/보 판단 및 저장
                gesture = classify_rps(hand_landmarks)
                detected_gestures[hand_label] = gesture

                # 화면 상의 손목 근처에 어떤 손인지/무슨 동작인지 텍스트 표시
                wrist_x, wrist_y = pixel_coords[0]
                cv2.putText(frame, f"{hand_label}: {gesture}", (wrist_x - 30, wrist_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ======================================================================
        # 4. 게임 화면 UI 및 오버레이 텍스트 출력
        # ======================================================================
        # 상단 스코어판
        cv2.putText(frame, f"Score - Left: {score_left} | Right: {score_right}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 현재 감지된 양손 상태
        cv2.putText(frame, f"Left: {detected_gestures['Left']} | Right: {detected_gestures['Right']}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 승패 결과 표시
        cv2.putText(frame, f"Result: {game_result}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 하단 조작 안내
        cv2.putText(frame, "Press 'Space' to Play 2-Player Game / 'q' or ESC to Exit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("2-Hand Rock Paper Scissors", frame)

        key = cv2.waitKey(5) & 0xFF
        
        # 스페이스바(ASCII 32) 입력 시 승패 판정
        if key == 32:
            print(f'key: {key}')
            left_choice = detected_gestures["Left"]
            right_choice = detected_gestures["Right"]

            # 두 손이 모두 올바르게 인식된 경우에만 게임 진행
            if left_choice in choices and right_choice in choices:
                if left_choice == right_choice:
                    game_result = f"Draw! ({left_choice} vs {right_choice})"
                
                # 왼손 승리 조건
                elif (left_choice == "가위" and right_choice == "보") or \
                    (left_choice == "바위" and right_choice == "가위") or \
                    (left_choice == "보" and right_choice == "바위"):
                    game_result = f"Left Hand Win! ({left_choice} vs {right_choice})"
                    score_left += 1
                
                # 오른손 승리 조건
                else:
                    game_result = f"Right Hand Win! ({left_choice} vs {right_choice})"
                    score_right += 1
            else:
                game_result = "Both hands must be visible & clear!"

        elif key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
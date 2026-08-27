# OpenCV 라이브러리: 카메라 영상 처리 및 화면 시각화 담당
import cv2

# MediaPipe 라이브러리: 얼굴 랜드마크 추론 모듈 제공
import mediapipe as mp

# time 라이브러리: MediaPipe Video 모드에 필요한 타임스탬프 계산용
import time

# ==============================================================================
# 1. MediaPipe Tasks API 필수 모듈 및 설정 로드
# ==============================================================================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Face Landmarker 세부 설정
options = FaceLandmarkerOptions(
    # 모델 파일 경로 지정 (.task 파일)
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    # 실시간 비디오 스트림 처리 모드 선택
    running_mode=VisionRunningMode.VIDEO,
    # 감지할 최대 얼굴 개수
    num_faces=2,
    # 얼굴 감지 최저 신뢰도 (50% 이상)
    min_face_detection_confidence=0.5,
    # 프레임 간 추적 최저 신뢰도 (50% 이상)
    min_tracking_confidence=0.5
)

# 0번 인덱스 기본 웹캠 연결
cap = cv2.VideoCapture(0)

# ==============================================================================
# 2. 실시간 얼굴 특징점 시각화 루프
# ==============================================================================
with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 거울 모드를 위한 좌우 반전
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # BGR 이미지를 RGB 이미지로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe 전용 Image 객체로 래핑
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 밀리초(ms) 단위 타임스탬프 계산
        frame_timestamp_ms = int(time.time() * 1000)

        # 얼굴 랜드마크 추론 실행
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # 얼굴이 감지되었을 경우 특징점 그리기
        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                # --------------------------------------------------------------
                # A. 468개 전체 입체 특징점(점) 찍기
                # --------------------------------------------------------------
                for lm in face_landmarks:
                    # 정규화된 좌표(0.0~1.0)를 실제 화면 픽셀 좌표로 환산
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # 관절 위치에 작은 초록색 원 그리기 (반지름: 1px)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

                # --------------------------------------------------------------
                # B. 주요 부위 강조 표시 (눈, 눈썹, 입술, 얼굴 윤곽 등)
                # --------------------------------------------------------------
                # 주요 랜드마크 인덱스 포인트 예시
                # 1번: 코 끝, 10번: 이마 중앙 상단, 152번: 턱 끝
                # 33번: 왼쪽 눈 가장자리, 263번: 오른쪽 눈 가장자리
                key_indices = [1, 10, 152, 33, 263, 61, 291]

                for idx in key_indices:
                    lm = face_landmarks[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # 핵심 특징점은 빨간색 큰 원으로 강조
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # 화면 안내 문구 표시
        cv2.putText(frame, "Face Landmarks Visualization", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' or ESC to exit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # 결과 프레임 출력
        cv2.imshow("Face Landmarks", frame)

        # 'q' 키 또는 ESC 키 입력 시 종료
        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):
            break

# ==============================================================================
# 3. 자원 해제
# ==============================================================================
# OpenCv 카메라 비디오 스트림 연결 해제
cap.release()

# 모든 OpenCV 그래픽 창 닫기
cv2.destroyAllWindows()

## 개발 진행 상황 (Development Log)

### **7월 1일 ~ 7월 4일**

*   **7/1 (화)**: 프로젝트 초기 설정 및 Git 저장소 생성. 기본 디렉터리 구조 설계.
*   **7/2 (수)**: '넘어짐 감지(Fall Detection)' 모델 학습을 위한 데이터셋 수집 및 전처리.
*   **7/3 (목)**: `model_train/train_in_local.py` 스크립트 작성 및 1차 모델(`model.pt`) 학습 진행. 
*   **7/4 (금)**: `yolo_with_rtsp` 프로젝트 구상 및 `crtsp` 디렉터리 구조 설정. OpenCV를 이용한 RTSP 스트림 연동 기능 개발 시작.

### **7월 7일 ~ 7월 11일**

*   **7/7 (월)**: **(오류 발생)** RTSP 스트림에서 간헐적으로 프레임 디코딩에 실패하여 프로그램이 비정상 종료되는 문제 발생.
*   **(해결)**  : 프레임 `read()` 이후, 프레임이 비어있는지 확인하는 `frame.empty()` 예외 처리 코드를 추가하여 안정성 확보.
*   **7/8 (화)**: yolo_with_rtsp/src/main.cpp`에 RTSP 스트림을 받아와 프레임 단위로 읽는 기능 구현 완료.
*   **7/9 (수)**: **(오류 발생)** `CMake` 빌드 시 TFLite 라이브러리 경로를 찾지 못하는 링커 오류 발생.
*   **(해결)** `CMakeLists.txt`의 `target_link_libraries` 경로를 절대 경로로 수정하고, 라이브러리 파일 권한 확인하여 문제 해결.
*   **7/10 (목)**: ONNX Runtime을 사용하여 `yolo11n-seg.onnx` 모델을 C++ 코드에 통합. 스트림에서 받아온 단일 프레임으로 객체 탐지 성공.
*   **7/11 (금)**: 실시간 탐지를 위해 `yolo_with_rtsp`의 탐지 로직을 무한 루프로 개선하고, 결과 출력 부분 구현.

### **7월 14일 ~ 7월 18일**

*   **7/14 (월)**: 하이퍼파라미터 튜닝 및 모델 재학습. 
*   **7/15 (화)**: 탐지된 객체에 바운딩 박스(Bounding Box)를 그리고, 처리된 영상을 화면에 출력하는 시각화 기능 추가.
*   **7/16 (수)**: `라즈베리파이 환경에 맞는 TFLite 라이브러리(`libtensorflowlite.so`)를 빌드하기 위해 **Bazel을 사용하여 TensorFlow 소스 코드를 직접 컴파일**. `raspi_tflite` C++ 프로젝트 설정 및 `CMakeLists.txt` 작성 시작.
*   **7/17 (목)**: `best_test` 모델 학습 및 TFLite 변환 작업 진행. `best_test_saved_model` 디렉터리에 결과물 저장.
*   **7/18 (금)**: 학습된 `fall.pt` 모델을 `export_model.py`를 사용해 ONNX, TFLite 형식으로 변환 완료.

### **7월 21일 ~ 7월 24일**

*   **7/21 (월)**: 모델 성능 최적화를 위해 Float16 및 Int8 양자화(Quantization) 진행. `fall_float16.tflite`, `fall_integer_quant.tflite` 모델 생성 및 성능 비교.
테스트 이미지로 라즈베리파이 환경에서 TFLite 모델 추론 성공 확인.
*   **7/22 (화)**: `raspi_tflite/main.cpp`에 이미지 파일을 입력받아 TFLite 모델로 추론하는 기본 로직 구현.
*   **7/23 (수)**: 
*   **7/24 (목)**: 


---

## 디렉터리 구조

```
AI-model/
├── model_train/         # AI 모델 학습, 변환, 저장 관련 디렉터리
│   ├── train_in_local.py    # 로컬 환경 모델 학습 스크립트
│   ├── export_model.py      # 모델을 TFLite, ONNX 등으로 변환하는 스크립트
│   ├── best_test_saved_model/ # 학습된 'best_test' 모델 및 변환된 파일 저장
│   └── fall_saved_model/      # 학습된 'fall' 모델 및 변환된 파일 저장
│
├── raspi_tflite/        # 라즈베리파이용 TFLite C++ 프로젝트
│   ├── main.cpp             # TFLite 모델 실행을 위한 메인 소스 코드
│   ├── CMakeLists.txt       # C++ 프로젝트 빌드를 위한 CMake 설정 파일
│   └── tensorflow_lite/     # TFLite C++ API 라이브러리
│
├── yolo_with_rtsp/      # YOLO 모델과 RTSP 스트림을 이용한 실시간 탐지 C++ 프로젝트
│   └── crtsp/
│       ├── src/main.cpp     # RTSP 스트림 처리 및 YOLO 모델 실행 코드
│       └── models/          # ONNX 형식의 YOLO 모델 저장
│
└── 
```

# CMakeLists.txt (통합 프로젝트용)

cmake_minimum_required(VERSION 3.10)
project(UnifiedStreamer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 필수 라이브러리 찾기 ---
find_package(OpenCV REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Threads REQUIRED)

# --- ONNX Runtime 라이브러리 경로 설정 ---
# 이 경로는 실제 다운로드하여 압축 푼 onnxruntime 폴더 경로로 수정해야 합니다.
set(ONNXRUNTIME_DIR "/home/a/onnxruntime-linux-aarch64-1.22.0") 

# --- [수정] 소스 파일 자동 검색 ---
# src 폴더와 그 하위 폴더에 있는 모든 .cpp 파일을 자동으로 찾아서 SOURCES 변수에 저장합니다.
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/yolo_backend/src/nn/*.cpp"
    "src/yolo_backend/src/utils/*.cpp"
)

# --- 실행 파일 생성 ---
# SOURCES 변수에 저장된 모든 소스 파일을 사용하여 실행 파일을 만듭니다.
add_executable(unified_streamer ${SOURCES})

# --- [수정] 헤더 검색 경로 추가 ---
# 헤더 파일(#include)을 찾을 폴더들을 지정합니다.
target_include_directories(unified_streamer PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
    "src"                       # detector.h, segmenter.h 등을 찾기 위해
    "src/yolo_backend/include"  # autobackend.h 등을 찾기 위해
)

# --- 라이브러리 링크 ---
# 필요한 라이브러리들을 실행 파일에 연결합니다.
target_link_libraries(unified_streamer PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
    SQLite::SQLite3
    Threads::Threads
)
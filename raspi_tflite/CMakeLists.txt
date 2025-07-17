cmake_minimum_required(VERSION 3.10)
project(RaspberryPiTFLiteDetection CXX)

# C++17 설정
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

# --- 수정된 부분 시작 ---

# PkgConfig를 사용하여 GStreamer와 OpenCV 라이브러리 검색
find_package(PkgConfig REQUIRED)
set(TFLITE_DIR "${CMAKE_SOURCE_DIR}/tensorflow_lite") 
include_directories(${TFLITE_DIR})
link_directories(${TFLITE_DIR}/lib)

# OpenCV 검색
pkg_check_modules(OpenCV REQUIRED opencv4)
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs:         ${OpenCV_LIBRARIES}")

# GStreamer 관련 모듈 검색
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-rtsp-server-1.0)
message(STATUS "GStreamer include dirs: ${GST_INCLUDE_DIRS}")
message(STATUS "GStreamer libs:         ${GST_LIBRARIES}")

# --- 수정된 부분 종료 ---


# TFLite 헤더들
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS} # GStreamer 헤더 경로 추가
    ${CMAKE_SOURCE_DIR}/include     # 워크스페이스 include 디렉터리
    ${TFLITE_DIR}/include 
)

# TFLite 라이브러리 경로 및 파일
set(TFLITE_LIB_DIR  "${CMAKE_SOURCE_DIR}/lib")
set(TFLITE_LIB_FILE "${TFLITE_LIB_DIR}/libtensorflowlite.so")
message(STATUS "TFLite lib: ${TFLITE_LIB_FILE}")

# 실행 파일
add_executable(raspi_detector main.cpp)

# 링크
target_link_libraries(raspi_detector
    ${OpenCV_LIBRARIES}     # OpenCV 라이브러리 변수명 변경
    ${GST_LIBRARIES}        # GStreamer 라이브러리 추가
    tensorflowlite
    pthread
    dl
)
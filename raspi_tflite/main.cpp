#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter_builder.h"

// XNNPACK 델리게이트를 위해 추가
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

// --- 설정 ---
// 모델 경로를 INT8 모델로 다시 설정합니다.
const std::string MODEL_PATH = "/home/a/raspi_tflite/best_test_full_integer_quant.tflite";
const std::vector<std::string> LABEL_MAP = {
    "person","ear","ear-mufs","face","face-guard","face-mask","foot",
    "tool","glasses","gloves","helmet","hands","head","medical-suit",
    "shoes","safety-suit","safety-vest"
};
const float CONF_THRESHOLD = 0.05f;
const float IOU_THRESHOLD  = 0.45f;
// -------------

// 모델의 입력/출력 양자화 파라미터를 전역 변수로 선언합니다.
// 이 값들은 모델의 실제 양자화 파라미터와 일치해야 합니다.
// 이전에 제공해주신 INT8 모델 정보를 바탕으로 설정합니다.
// Input 0: quant=(0.003921568859368563, -128)
// Output 0: quant=(0.00812158640474081, -105)
float INPUT_SCALE = 0.003921568859368563f;
int INPUT_ZERO_POINT = -128;

float OUTPUT_SCALE = 0.00812158640474081f;
int OUTPUT_ZERO_POINT = -105;

int main() {
    // 1) TFLite 모델 로드 및 Interpreter 생성
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(MODEL_PATH.c_str());
    if (!model) {
        std::cerr << "모델 로드 실패: " << MODEL_PATH << std::endl;
        return -1;  
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter) {
        std::cerr << "인터프리터 생성 실패" << std::endl;
        return -1;
    }

    // --- XNNPACK 델리게이트 추가 ---
    TfLiteXNNPackDelegateOptions xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_options.num_threads = 4; // 라즈베리파이 코어 수에 맞게 설정
    std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
        xnnpack_delegate(TfLiteXNNPackDelegateCreate(&xnnpack_options), &TfLiteXNNPackDelegateDelete);

    if (interpreter->ModifyGraphWithDelegate(xnnpack_delegate.get()) != kTfLiteOk) {
        std::cerr << "XNNPACK 델리게이트 추가 실패! (모델 양자화 호환성 문제일 수 있음)" << std::endl;
        // 델리게이트 추가 실패해도 프로그램은 계속 실행됩니다 (CPU 폴백).
    } else {
        std::cout << "XNNPACK 델리게이트가 성공적으로 추가되었습니다." << std::endl;
    }
    // ----------------------------

    interpreter->AllocateTensors();

    // 입력 텐서 정보
    int input_idx = interpreter->inputs()[0];
    TfLiteIntArray* in_dims = interpreter->tensor(input_idx)->dims;
    const int IN_H = in_dims->data[1];
    const int IN_W = in_dims->data[2];
    const int IN_C = in_dims->data[3];

    // 입력 텐서의 실제 양자화 파라미터를 읽어옵니다. (코드의 전역변수와 일치하는지 확인용)
    // --- 오류 수정 부분 ---
    // interpreter->tensor(input_idx)->params에서 바로 scale과 zero_point에 접근합니다.
    INPUT_SCALE = interpreter->tensor(input_idx)->params.scale;
    INPUT_ZERO_POINT = interpreter->tensor(input_idx)->params.zero_point;
    // ---------------------
    std::cout << "모델 입력 크기: " << IN_W << "x" << IN_H << "x" << IN_C
              << ", DType: INT8, Quant: (S=" << INPUT_SCALE << ", ZP=" << INPUT_ZERO_POINT << ")" << std::endl;

    // 출력 텐서 정보
    int output_idx = interpreter->outputs()[0];
    TfLiteIntArray* out_dims = interpreter->tensor(output_idx)->dims;
    const int NUM_ATTRIBUTES = out_dims->data[1]; // 21
    const int NUM_DETECTIONS = out_dims->data[2]; // 2100

    // 출력 텐서의 실제 양자화 파라미터를 읽어옵니다.
    // --- 오류 수정 부분 ---
    // interpreter->tensor(output_idx)->params에서 바로 scale과 zero_point에 접근합니다.
    OUTPUT_SCALE = interpreter->tensor(output_idx)->params.scale;
    OUTPUT_ZERO_POINT = interpreter->tensor(output_idx)->params.zero_point;
    // ---------------------
    std::cout << "모델 출력 Shape: [1, " << NUM_ATTRIBUTES << ", " << NUM_DETECTIONS
              << "], DType: INT8, Quant: (S=" << OUTPUT_SCALE << ", ZP=" << OUTPUT_ZERO_POINT << ")" << std::endl;


    // 2) 카메라 열기
    cv::VideoCapture cap(
        "libcamerasrc ! video/x-raw,width=320,height=320,format=RGB ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink",
        cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "카메라 열기 실패. GStreamer 및 libcamera 설정 확인 필요." << std::endl;
        return -1;
    }

    cv::Mat frame;
    // FPS 계산을 위한 초기 시간
    auto last_frame_time = std::chrono::high_resolution_clock::now();

    while (true) {
        auto t0 = std::chrono::high_resolution_clock::now(); // 프레임 처리 시작 시간
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "프레임을 읽을 수 없습니다. 스트림 종료." << std::endl;
            break;
        }
        int H0 = frame.rows, W0 = frame.cols;

        // 3) 전처리: 리사이즈 -> BGR2RGB -> float32 정규화 -> INT8 양자화
        cv::Mat img;
        cv::resize(frame, img, cv::Size(IN_W, IN_H));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // INT8 모델 입력을 위한 양자화
        // 0-255 범위의 픽셀 값을 INT8로 변환 (float = (int8 - zero_point) * scale)
        // int8 = round(float / scale + zero_point)
        // img (CV_8UC3) -> input_tensor (int8)
        std::vector<int8_t> input_tensor(IN_H * IN_W * IN_C);
        for (int i = 0; i < IN_H * IN_W * IN_C; ++i) {
            float pixel_val = static_cast<float>(img.data[i]); // CV_8UC3의 픽셀 값 (0-255)
            // Python의 (np.float32)/255.0과 동일한 정규화 후 양자화
            float normalized_float = pixel_val / 255.0f;
            input_tensor[i] = static_cast<int8_t>(std::round(normalized_float / INPUT_SCALE + INPUT_ZERO_POINT));
            // 클리핑 (INT8 범위 -128 ~ 127)
            if (input_tensor[i] > 127) input_tensor[i] = 127;
            if (input_tensor[i] < -128) input_tensor[i] = -128;
        }


        // 4) 입력 버퍼에 복사
        int8_t* input_ptr = interpreter->typed_tensor<int8_t>(input_idx); // DType을 int8_t로 변경
        std::memcpy(input_ptr, input_tensor.data(), IN_H * IN_W * IN_C * sizeof(int8_t));

        // 5) 추론
        interpreter->Invoke();
        int8_t* out_data_int8 = interpreter->typed_tensor<int8_t>(output_idx); // DType을 int8_t로 변경

        // 6) 후처리 (Detection Box 추출)
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> classes;

        for (int i = 0; i < NUM_DETECTIONS; ++i) {
            // INT8 출력값을 float으로 역양자화 (dequantize)
            float cx = (static_cast<float>(out_data_int8[0 * NUM_DETECTIONS + i]) - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
            float cy = (static_cast<float>(out_data_int8[1 * NUM_DETECTIONS + i]) - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
            float w  = (static_cast<float>(out_data_int8[2 * NUM_DETECTIONS + i]) - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
            float h  = (static_cast<float>(out_data_int8[3 * NUM_DETECTIONS + i]) - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;

            float best_conf = 0.0f;
            int class_id = -1;
            for (int k = 0; k < LABEL_MAP.size(); ++k) {
                // 클래스 신뢰도도 역양자화
                float current_conf = (static_cast<float>(out_data_int8[(4 + k) * NUM_DETECTIONS + i]) - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
                if (current_conf > best_conf) {
                    best_conf = current_conf;
                    class_id = k;
                }
            }
            
            if (best_conf < CONF_THRESHOLD) continue;

            // 바운딩 박스 만들기 (cx, cy, w, h가 0-1 사이로 정규화된 값이라고 가정)
            int x1 = static_cast<int>(round((cx - w/2) * W0));
            int y1 = static_cast<int>(round((cy - h/2) * H0));
            int w_box = static_cast<int>(round(w * W0));
            int h_box = static_cast<int>(round(h * H0));

            // 클리핑
            x1 = std::max(0, std::min(x1, W0 - 1));
            y1 = std::max(0, std::min(y1, H0 - 1));
            int x2 = std::max(x1 + 1, std::min(x1 + w_box, W0));
            int y2 = std::max(y1 + 1, std::min(y1 + h_box, H0));
            
            boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
            scores.push_back(best_conf);
            classes.push_back(class_id);
        }

        // 7) NMS 적용
        std::vector<int> nms_idx;
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD, nms_idx);
        }

        // 8) 결과 그리기
        for (int idx : nms_idx) {
            cv::Rect box = boxes[idx];
            std::string label;
            if (classes[idx] >= 0 && classes[idx] < LABEL_MAP.size()) {
                label = LABEL_MAP[classes[idx]] + ": " + cv::format("%.2f", scores[idx]);
            } else {
                label = "Unknown: " + cv::format("%.2f", scores[idx]);
            }
            
            cv::rectangle(frame, box, cv::Scalar(0,255,0), 2);

            cv::Point text_origin(box.x, box.y - 5);
            if (text_origin.y < 15) text_origin.y = box.y + 15;

            cv::putText(frame, label, text_origin,
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
        }

        // 9) FPS 계산 및 화면 출력
        auto current_frame_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed_time = current_frame_time - last_frame_time;
        last_frame_time = current_frame_time;
        float fps = 1.0f / elapsed_time.count();

        cv::putText(frame, cv::format("FPS: %.1f", fps), cv::Point(20,40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,255), 2);
        cv::imshow("YOLOv11 INT8 Debug (C++)", frame);

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
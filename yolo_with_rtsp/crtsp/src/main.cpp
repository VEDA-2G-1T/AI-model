// main.cpp (모자이크 효과 적용)

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <csignal>
#include <ctime>
#include <numeric>
#include <algorithm>
#include <set>
#include <memory>

#include <opencv2/opencv.hpp>
#include <sqlite3.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "detector.h"
#include "segmenter.h"

// --- 전역 변수 및 설정 ---
std::atomic<bool> keep_running(true);
std::string current_mode = "blur"; // 테스트를 위해 기본 모드 변경
std::mutex mode_mutex;
const char* SOCKET_FILE = "/tmp/streaming_control.sock";
const char* DETECTION_DB_FILE = "detections.db";
const char* BLUR_DB_FILE = "blur.db";
const char* IMAGE_SAVE_DIR = "captured_images";

// --- 함수 선언 ---
void signal_handler(int signum);
void run_control_server();
FILE* create_ffmpeg_process(const std::string& rtsp_url);
void save_detection_log(const std::vector<DetectionResult>& results, const cv::Mat& frame, const Detector& detector);
void save_blur_log(int person_count);
std::string get_current_timestamp();
void init_directories_and_databases();

std::string gstreamer_pipeline (int capture_width, int capture_height, int framerate) {
    return "libcamerasrc ! video/x-raw, width=" + std::to_string(capture_width) + ", height=" + std::to_string(capture_height) + ", framerate=" + std::to_string(framerate) + "/1 ! videoconvert ! appsink";
}

// --- 메인 함수 ---
int main() {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    init_directories_and_databases();

    std::unique_ptr<Detector> detector = nullptr;
    std::unique_ptr<Segmenter> segmenter = nullptr;
    std::string last_loaded_mode = "none";

    std::thread control_thread(run_control_server);

    int capture_width = 320;
    int capture_height = 240;
    int framerate = 15;
    std::string pipeline = gstreamer_pipeline(capture_width, capture_height, framerate);
    std::cout << "GStreamer Pipeline: " << pipeline << std::endl;

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "오류: 카메라를 열 수 없습니다. GStreamer 또는 카메라 설정을 확인하세요." << std::endl;
        return -1;
    }
    std::cout << "카메라 설정 완료." << std::endl;

    std::string rtsp_url_processed = "rtsps://127.0.0.1:8322/processed";
    std::string rtsp_url_raw = "rtsps://127.0.0.1:8322/raw";
    FILE* proc_processed = create_ffmpeg_process(rtsp_url_processed);
    FILE* proc_raw = create_ffmpeg_process(rtsp_url_raw);
    std::cout << "스트리밍 시작... 메인 루프에 진입합니다." << std::endl;

    cv::Mat frame;
    int frame_counter = 0;
    time_t last_save_time = time(0);

    while (keep_running) {
        if (!cap.read(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        if (proc_raw) {
            fwrite(frame.data, 1, frame.total() * frame.elemSize(), proc_raw);
            fflush(proc_raw);
        }

        std::string active_mode;
        {
            std::lock_guard<std::mutex> lock(mode_mutex);
            active_mode = current_mode;
        }
        
        if (active_mode != last_loaded_mode) {
            detector.reset();
            segmenter.reset();
            std::cout << "모드 변경: 기존 모델 메모리 해제" << std::endl;
            
            if (active_mode == "detect") {
                detector = std::make_unique<Detector>();
                std::cout << "Detector 모델 로드 완료." << std::endl;

            // ================== [수정 1] Segmenter 생성 부분 ==================
            } else if (active_mode == "blur") {
                // 모델 경로만 인자로 전달하여 새로운 Segmenter를 생성합니다.
                std::string model_path = "/home/a/crtsp/models/yolov8n-seg-320.onnx"; // 사용할 YOLOv8 모델 경로
                segmenter = std::make_unique<Segmenter>(model_path);
                std::cout << "Segmenter 모델 로드 완료." << std::endl;
            }
            // =================================================================
            last_loaded_mode = active_mode;
        }
        cv::Mat display_frame = frame.clone();

        if (active_mode == "stop") {
             cv::putText(display_frame, "STOPPED", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        } else if (frame_counter % 3 == 0) {
            if (active_mode == "detect" && detector) {
                // ... (detect 모드 로직은 그대로) ...
            
            // ================== [수정 2] blur 모드 처리 부분 ==================
            } else if (active_mode == "blur" && segmenter) {
                // 1. process_frame 함수를 호출하여 display_frame에 직접 블러 처리를 적용합니다.
                //    (이제 이 함수는 아무것도 반환하지 않습니다. 로깅은 아래에서 별도 처리)
                segmenter->process_frame(display_frame);

                // 2. 데이터베이스 로깅을 위한 사람 수 카운트는 필요 시 별도 함수로 구현하거나,
                //    process_frame 함수가 사람 수를 반환하도록 수정할 수 있습니다.
                //    (지금은 로깅 기능을 잠시 비활성화합니다.)
                /*
                if (time(0) - last_save_time >= 3) {
                    // int person_count = segmenter->get_person_count(); // 이런 함수를 추가해야 함
                    // save_blur_log(person_count);
                    last_save_time = time(0);
                }
                */
               
                // 3. 블러 처리가 함수 내부에서 모두 끝났으므로,
                //    기존의 마스크를 빨갛게 칠하는 코드는 필요 없습니다.
            }
            // =================================================================
        }
        
        cv::putText(display_frame, "MODE: " + active_mode, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);


        if (active_mode == "stop") {
             cv::putText(display_frame, "STOPPED", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        } else if (frame_counter % 3 == 0) {
            if (active_mode == "detect" && detector) {
                auto results = detector->detect(frame, 0.5, 0.45);
                if (time(0) - last_save_time >= 3) {
                    save_detection_log(results, frame, *detector);
                    last_save_time = time(0);
                }
                for(const auto& res : results) {
                    cv::rectangle(display_frame, res.box, cv::Scalar(0, 255, 0), 2);
                }
            } else if (active_mode == "blur" && segmenter) {
                // segmenter->process_frame 함수를 호출하고, 반환된 구조체에서 사람 수를 얻습니다.
                SegmentationResult result = segmenter->process_frame(display_frame);

                if (time(0) - last_save_time >= 3) {
                    save_blur_log(result.person_count); // 반환된 값으로 로그 저장
                    last_save_time = time(0);
                }

                // 블러 처리는 process_frame 함수 내부에서 이미 완료되었습니다.
                // 따라서 display_frame.setTo(...) 라인은 필요 없습니다.
            }
        }
        
        cv::putText(display_frame, "MODE: " + active_mode, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

        if (proc_processed) {
            fwrite(display_frame.data, 1, display_frame.total() * display_frame.elemSize(), proc_processed);
            fflush(proc_processed);
        }
        frame_counter++;
    }

    std::cout << "정리 중..." << std::endl;
    if (proc_processed) pclose(proc_processed);
    if (proc_raw) pclose(proc_raw);
    keep_running = false;
    if (control_thread.joinable()) control_thread.join();
    cap.release();

    return 0;
}

// --- 나머지 함수들은 이전과 동일 ---
void signal_handler(int signum) {
    std::cout << "종료 신호 (" << signum << ") 수신." << std::endl;
    keep_running = false;
}

void run_control_server() {
    unlink(SOCKET_FILE);
    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd == -1) { perror("socket"); return; }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_FILE, sizeof(addr.sun_path) - 1);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) { perror("bind"); return; }
    if (listen(server_fd, 5) == -1) { perror("listen"); return; }
    std::cout << "✅ 제어 소켓 서버가 " << SOCKET_FILE << " 에서 실행 중입니다." << std::endl;

    while (keep_running) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd == -1) continue;

        char buf[256];
        int n = read(client_fd, buf, 255);
        if (n > 0) {
            buf[n] = '\0';
            std::string cmd(buf);
            size_t pos_start = cmd.find(":") + 2;
            size_t pos_end = cmd.find("\"", pos_start);
            if (pos_start != std::string::npos && pos_end != std::string::npos) {
                std::string mode = cmd.substr(pos_start, pos_end - pos_start);
                std::lock_guard<std::mutex> lock(mode_mutex);
                current_mode = mode;
                std::cout << "✅ 제어 명령 수신: 모드를 '" << current_mode << "'로 변경합니다." << std::endl;
            }
        }
        close(client_fd);
    }
    close(server_fd);
    unlink(SOCKET_FILE);
}

FILE* create_ffmpeg_process(const std::string& rtsp_url) {
    std::string cmd = "ffmpeg -f rawvideo -pixel_format bgr24 -video_size 320x240 -framerate 15 -i - "
                      "-c:v h264_v4l2m2m -b:v 2M -bufsize 2M -pix_fmt yuv420p "
                      "-f rtsp -rtsp_transport tcp " + rtsp_url;
    return popen(cmd.c_str(), "w");
}

void save_detection_log(const std::vector<DetectionResult>& results, const cv::Mat& frame, const Detector& detector) {
    if (results.empty()) return;

    int person_count = 0, helmet_count = 0, safety_vest_count = 0;
    double total_confidence = 0;
    std::set<std::string> unique_objects;

    const auto& class_names = detector.get_class_names();
    for(const auto& res : results) {
        std::string class_name = class_names[res.class_id];
        if (class_name == "person") person_count++;
        else if (class_name == "helmet") helmet_count++;
        else if (class_name == "safety-vest") safety_vest_count++;
        unique_objects.insert(class_name);
        total_confidence += res.confidence;
    }

    std::string timestamp_str = get_current_timestamp();
    std::string image_path = "";
    bool is_normal_state = (helmet_count == safety_vest_count) && (person_count <= helmet_count);
    if (!is_normal_state) {
        std::string timestamp_file = timestamp_str;
        std::replace(timestamp_file.begin(), timestamp_file.end(), ':', '-');
        std::replace(timestamp_file.begin(), timestamp_file.end(), ' ', '_');
        image_path = std::string(IMAGE_SAVE_DIR) + "/" + timestamp_file + ".jpg";
        cv::imwrite(image_path, frame);
    }

    std::string all_objects_str;
    for(const auto& obj : unique_objects) { all_objects_str += obj + ", "; }
    if (!all_objects_str.empty()) all_objects_str.resize(all_objects_str.length() - 2);

    double avg_confidence = results.empty() ? 0.0 : total_confidence / results.size();

    sqlite3* db;
    if (sqlite3_open(DETECTION_DB_FILE, &db) == SQLITE_OK) {
        sqlite3_exec(db, "PRAGMA journal_mode=WAL;", NULL, NULL, NULL);
        const char* sql = "INSERT INTO detections VALUES(?,?,?,?,?,?,?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, 0) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, timestamp_str.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, all_objects_str.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(stmt, 3, person_count);
            sqlite3_bind_int(stmt, 4, helmet_count);
            sqlite3_bind_int(stmt, 5, safety_vest_count);
            sqlite3_bind_double(stmt, 6, avg_confidence);
            sqlite3_bind_text(stmt, 7, image_path.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
        }
        sqlite3_finalize(stmt);
        sqlite3_close(db);
    }
}

void save_blur_log(int person_count) {
    sqlite3* db;
    if (sqlite3_open(BLUR_DB_FILE, &db) == SQLITE_OK) {
        sqlite3_exec(db, "PRAGMA journal_mode=WAL;", NULL, NULL, NULL);
        const char* sql = "INSERT INTO person_counts VALUES(?,?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, 0) == SQLITE_OK) {
            std::string timestamp_str = get_current_timestamp();
            sqlite3_bind_text(stmt, 1, timestamp_str.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(stmt, 2, person_count);
            sqlite3_step(stmt);
        }
        sqlite3_finalize(stmt);
        sqlite3_close(db);
    }
}

std::string get_current_timestamp() {
    time_t now = time(0);
    char buf[80];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));
    return buf;
}

void init_directories_and_databases() {
    mkdir(IMAGE_SAVE_DIR, 0777);

    sqlite3* db_det;
    if (sqlite3_open(DETECTION_DB_FILE, &db_det) == SQLITE_OK) {
        const char* sql = "CREATE TABLE IF NOT EXISTS detections ("
                          "timestamp TEXT NOT NULL, all_objects TEXT, person_count INTEGER NOT NULL, "
                          "helmet_count INTEGER NOT NULL, safety_vest_count INTEGER NOT NULL, "
                          "avg_confidence REAL, image_path TEXT);";
        sqlite3_exec(db_det, sql, 0, 0, 0);
        sqlite3_close(db_det);
    }
    sqlite3* db_blur;
    if (sqlite3_open(BLUR_DB_FILE, &db_blur) == SQLITE_OK) {
        const char* sql = "CREATE TABLE IF NOT EXISTS person_counts ("
                          "timestamp TEXT NOT NULL, count INTEGER NOT NULL);";
        sqlite3_exec(db_blur, sql, 0, 0, 0);
        sqlite3_close(db_blur);
    }
}

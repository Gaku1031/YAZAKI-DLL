#include "rppg.h"
#include "peak_detect.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <vector>
#include <filesystem>
#include <chrono>

// OpenCV DNN for face detection
#include <opencv2/dnn.hpp>

// 顔のROIのランドマーク番号（MediaPipeの顔メッシュ）
const std::vector<int> FACE_ROI_LANDMARKS = {
    118, 119, 120, 47, 217, 174, 196, 197, 419, 399, 437, 277,
    349, 348, 347, 280, 425, 426, 391, 393, 164, 167, 165, 206, 205, 50, 118
};

// 詳細タイミング計測用の構造体
struct RPPGTiming {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    std::string stage_name;
    
    void start(const std::string& name) {
        stage_name = name;
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    void end() {
        end_time = std::chrono::high_resolution_clock::now();
    }
    
    double get_duration_ms() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
};

struct RPPGProcessor::Impl {
    // OpenCV DNN Face Detection
    cv::dnn::Net face_detector;
    bool dnn_initialized = false;
    std::string model_dir;
    
    // タイミング記録
    std::vector<RPPGTiming> timing_log;
    
    Impl(const std::string& model_directory = "") : model_dir(model_directory) {
        // Initialize OpenCV DNN face detector
        try {
            std::string pb_path = "opencv_face_detector_uint8.pb";
            std::string pbtxt_path = "opencv_face_detector.pbtxt";
            
            // Try multiple paths for OpenCV DNN files
            std::vector<std::string> pb_paths = {
                pb_path,  // Current directory
                model_dir + "/" + pb_path,  // Model directory
                "models/" + pb_path,  // Models subdirectory
                "../models/" + pb_path  // Parent models directory
            };
            
            std::vector<std::string> pbtxt_paths = {
                pbtxt_path,  // Current directory
                model_dir + "/" + pbtxt_path,  // Model directory
                "models/" + pbtxt_path,  // Models subdirectory
                "../models/" + pbtxt_path  // Parent models directory
            };
            
            bool pb_found = false;
            bool pbtxt_found = false;
            std::string final_pb_path, final_pbtxt_path;
            
            // Find pb file
            for (const auto& path : pb_paths) {
                std::ifstream f(path, std::ios::binary);
                if (f.good()) {
                    final_pb_path = path;
                    pb_found = true;
                    std::cout << "[RPPG] Found OpenCV DNN pb file: " << path << std::endl;
                    break;
                }
            }
            
            // Find pbtxt file
            for (const auto& path : pbtxt_paths) {
                std::ifstream f(path, std::ios::binary);
                if (f.good()) {
                    final_pbtxt_path = path;
                    pbtxt_found = true;
                    std::cout << "[RPPG] Found OpenCV DNN pbtxt file: " << path << std::endl;
                    break;
                }
            }
            
            if (!pb_found || !pbtxt_found) {
                std::cerr << "[RPPG] OpenCV DNN files not found. Tried paths:" << std::endl;
                for (const auto& path : pb_paths) {
                    std::cerr << "  PB: " << path << std::endl;
                }
                for (const auto& path : pbtxt_paths) {
                    std::cerr << "  PBTXT: " << path << std::endl;
                }
                dnn_initialized = false;
                return;
            }
            
            face_detector = cv::dnn::readNetFromTensorflow(final_pb_path, final_pbtxt_path);
            dnn_initialized = true;
            std::cout << "[RPPG] OpenCV DNN face detector initialized successfully" << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "Failed to load OpenCV DNN face detector: " << e.what() << std::endl;
            dnn_initialized = false;
        }
    }
    
    // タイミング計測ヘルパー関数
    void start_timing(const std::string& stage_name) {
        RPPGTiming timing;
        timing.start(stage_name);
        timing_log.push_back(timing);
    }
    
    void end_timing() {
        if (!timing_log.empty()) {
            timing_log.back().end();
        }
    }
    
    // タイミングログをクリア
    void clear_timing_log() {
        timing_log.clear();
    }
    
    std::string get_timing_summary() const {
        std::stringstream ss;
        ss << "\n=== RPPG TIMING ANALYSIS ===\n";
        
        printf("[DEBUG] RPPG timing_log size: %zu\n", timing_log.size());
        
        double total_time = 0.0;
        for (const auto& timing : timing_log) {
            double duration = timing.get_duration_ms();
            total_time += duration;
            ss << std::fixed << std::setprecision(2) 
               << timing.stage_name << ": " << duration << " ms\n";
            printf("[DEBUG] RPPG stage: %s = %.2f ms\n", timing.stage_name.c_str(), duration);
        }
        
        ss << "Total RPPG time: " << total_time << " ms\n";
        ss << "=== RPPG BREAKDOWN ===\n";
        
        // 各段階の割合を計算
        for (const auto& timing : timing_log) {
            double duration = timing.get_duration_ms();
            double percentage = (total_time > 0) ? (duration / total_time) * 100.0 : 0.0;
            ss << std::fixed << std::setprecision(1) 
               << timing.stage_name << ": " << percentage << "%\n";
        }
        
        std::string result = ss.str();
        printf("[DEBUG] RPPG timing summary length: %zu\n", result.length());
        return result;
    }
    
    std::vector<cv::Point2f> detectFaceLandmarks(const cv::Mat& frame) {
        std::vector<cv::Point2f> landmarks;
        
        if (!dnn_initialized) {
            std::cerr << "[RPPG] Face detector not initialized" << std::endl;
            return landmarks;
        }
        
        // Prepare input blob
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0), false, false);
        face_detector.setInput(blob);
        
        // Forward pass
        cv::Mat detections = face_detector.forward();
        
        // Process detections
        cv::Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
        
        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);
            
            if (confidence > 0.5) { // Confidence threshold
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
                
                // Create face ROI landmarks (rectangular)
                landmarks.push_back(cv::Point2f(x1, y1));
                landmarks.push_back(cv::Point2f(x2, y1));
                landmarks.push_back(cv::Point2f(x2, y2));
                landmarks.push_back(cv::Point2f(x1, y2));
                
                // Add more points for better ROI coverage
                landmarks.push_back(cv::Point2f((x1 + x2) / 2, y1));
                landmarks.push_back(cv::Point2f((x1 + x2) / 2, y2));
                landmarks.push_back(cv::Point2f(x1, (y1 + y2) / 2));
                landmarks.push_back(cv::Point2f(x2, (y1 + y2) / 2));
                
                break; // Use first detected face
            }
        }
        
        return landmarks;
    }
};

RPPGProcessor::RPPGProcessor(const std::string& model_dir) : pImpl(std::make_unique<Impl>(model_dir)) {
    // OpenCV DNN Face Detectionは既にImplコンストラクタで初期化済み
}

RPPGProcessor::~RPPGProcessor() = default;

std::string RPPGProcessor::get_timing_summary() const {
    return pImpl->get_timing_summary();
}

RPPGResult RPPGProcessor::processVideo(const std::string& videoPath) {
    namespace fs = std::filesystem;
    std::string temp_dir = "temp_frames_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    fs::create_directory(temp_dir);

    // 2. ffmpegで画像シーケンスに変換（Windowsでは .\ffmpeg.exe を優先）
    std::string ffmpeg_bin = "ffmpeg";
#ifdef _WIN32
    ffmpeg_bin = ".\\ffmpeg.exe";
#endif
    std::string ffmpeg_log = temp_dir + "/ffmpeg_out.log";
    std::string ffmpeg_cmd = ffmpeg_bin + " -y -i \"" + videoPath + "\" -q:v 2 \"" + temp_dir + "/frame_%05d.jpg\" > \"" + ffmpeg_log + "\" 2>&1";
    int ret = system(ffmpeg_cmd.c_str());
    if (ret != 0) {
        // ffmpegのログ内容をdll_error.logに転記
        std::ifstream flog(ffmpeg_log);
        std::ofstream err("dll_error.log", std::ios::app);
        err << "[ffmpeg error log]" << std::endl;
        if (flog) err << flog.rdbuf() << std::endl;
        err << "[ffmpeg command] " << ffmpeg_cmd << std::endl;
        fs::remove_all(temp_dir);
        throw std::runtime_error("ffmpeg failed to extract frames from video: " + videoPath);
    }

    // 3. 画像ファイルリストを取得
    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::directory_iterator(temp_dir)) {
        if (entry.is_regular_file()) {
            imagePaths.push_back(entry.path().string());
        }
    }
    std::sort(imagePaths.begin(), imagePaths.end());
    if (imagePaths.empty()) {
        fs::remove_all(temp_dir);
        throw std::runtime_error("No frames extracted from video: " + videoPath);
    }

    // 4. 画像シーケンスで解析
    double fps = 30.0; // 必要ならffmpegでfps取得も可
    RPPGResult result = processImagesFromPaths(imagePaths, fps);

    // 5. 一時ファイル削除
    fs::remove_all(temp_dir);

    return result;
}

RPPGResult RPPGProcessor::processImagesFromPaths(const std::vector<std::string>& imagePaths, double fps) {
    // タイミングログをクリア
    pImpl->clear_timing_log();
    
    pImpl->start_timing("RPPG Total Processing");
    
    std::vector<std::vector<double>> skin_means;
    std::vector<double> timestamps;
    int frame_count = 0;
    
    pImpl->start_timing("Image Loading and Processing");
    
    for (const auto& path : imagePaths) {
        cv::Mat frame = cv::imread(path);
        if (frame.empty()) continue;
        double current_time = frame_count / fps;
        frame_count++;
        
        // 顔検出とROI抽出
        std::vector<cv::Point2f> landmarks = pImpl->detectFaceLandmarks(frame);
        if (!landmarks.empty()) {
            // ROIマスクの作成
            cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
            std::vector<cv::Point> roi_points;
            for (const auto& pt : landmarks) {
                roi_points.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
            }
            if (roi_points.size() >= 3) {
                cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{roi_points}, cv::Scalar(255));
                
                // 肌色フィルタリング（YCbCr色空間）
                cv::Mat frame_ycbcr;
                cv::cvtColor(frame, frame_ycbcr, cv::COLOR_BGR2YCrCb);
                cv::Mat skin_mask;
                cv::inRange(frame_ycbcr, cv::Scalar(0, 100, 130), cv::Scalar(255, 140, 175), skin_mask);
                
                // ROIマスクと肌色マスクの組み合わせ
                cv::Mat combined_mask;
                cv::bitwise_and(mask, skin_mask, combined_mask);
                
                // ROI領域の色平均を計算
                cv::Scalar mean = cv::mean(frame, combined_mask);
                // 有効な平均値のみを保存
                if (mean[0] > 0 && mean[1] > 0 && mean[2] > 0) {
                    skin_means.push_back({mean[0], mean[1], mean[2]});
                    timestamps.push_back(current_time);
                }
            }
        }
    }
    pImpl->end_timing(); // Image Loading and Processing
    
    pImpl->start_timing("Signal Processing");
    std::vector<double> rppg_signal, time_data, peak_times;
    if (skin_means.size() > 30) {
        pImpl->start_timing("Eigen Matrix Operations");
        int L = 30;
        int frames = static_cast<int>(skin_means.size());
        Eigen::MatrixXd C(3, frames);
        for (int i = 0; i < frames; ++i) {
            C(0, i) = skin_means[i][0];
            C(1, i) = skin_means[i][1];
            C(2, i) = skin_means[i][2];
        }
        pImpl->end_timing();
        
        pImpl->start_timing("RPPG Algorithm");
        Eigen::VectorXd H = Eigen::VectorXd::Zero(frames);
        for (int f = 0; f < frames - L + 1; ++f) {
            Eigen::MatrixXd block = C.block(0, f, 3, L);
            Eigen::Vector3d mu_C = block.rowwise().mean();
            for (int i = 0; i < 3; ++i) {
                if (mu_C(i) == 0) mu_C(i) = 1e-8;
            }
            Eigen::MatrixXd C_normed = block.array().colwise() / mu_C.array();
            Eigen::MatrixXd M(2, 3);
            M << 0, 1, -1,
                -2, 1, 1;
            Eigen::MatrixXd S = M * C_normed;
            double alpha = std::sqrt(S.row(0).array().square().mean()) /
                          (std::sqrt(S.row(1).array().square().mean()) + 1e-8);
            Eigen::VectorXd P = S.row(0).transpose() + alpha * S.row(1).transpose();
            double P_mean = P.mean();
            double P_std = std::max(std::sqrt((P.array() - P_mean).square().mean()), 1e-8);
            Eigen::VectorXd P_normalized = (P.array() - P_mean) / P_std;
            H.segment(f, L) += P_normalized;
        }
        pImpl->end_timing();
        
        pImpl->start_timing("Signal Normalization");
        double mean = H.mean();
        double stddev = std::sqrt((H.array() - mean).square().sum() / H.size());
        Eigen::VectorXd pulse_z = (H.array() - mean) / (stddev + 1e-8);
        rppg_signal.assign(pulse_z.data(), pulse_z.data() + pulse_z.size());
        time_data = timestamps;
        pImpl->end_timing();
        
        pImpl->start_timing("Peak Detection");
        std::vector<size_t> peaks = find_peaks(rppg_signal, 10, 0.0);
        for (size_t idx : peaks) {
            if (idx < time_data.size()) {
                peak_times.push_back(time_data[idx]);
            }
        }
        pImpl->end_timing();
    }
    pImpl->end_timing(); // Signal Processing
    
    pImpl->end_timing(); // RPPG Total Processing
    
    RPPGResult result;
    result.rppg_signal = rppg_signal;
    result.time_data = time_data;
    result.peak_times = peak_times;
    return result;
} 

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
#include <sstream>
#include <iomanip>

// OpenCV DNN for face detection
#include <opencv2/dnn.hpp>

// 顔のROIのランドマーク番号（MediaPipeの顔メッシュ）
const std::vector<int> FACE_ROI_LANDMARKS = {
    118, 119, 120, 47, 217, 174, 196, 197, 419, 399, 437, 277,
    349, 348, 347, 280, 425, 426, 391, 393, 164, 167, 165, 206, 205, 50, 118
};

// RPPGTiming構造体のメンバー関数実装
RPPGTiming::RPPGTiming() : is_active(false) {}

void RPPGTiming::start(const std::string& name) {
    stage_name = name;
    start_time = std::chrono::high_resolution_clock::now();
    is_active = true;
}

void RPPGTiming::end() {
    if (is_active) {
        end_time = std::chrono::high_resolution_clock::now();
        is_active = false;
    }
}

double RPPGTiming::get_duration_ms() const {
    if (!is_active) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
    return 0.0;
}

// 詳細タイミング計測用の構造体
struct RPPGProcessor::Impl {
    // OpenCV DNN Face Detection
    cv::dnn::Net face_detector;
    bool dnn_initialized = false;
    std::string model_dir;
    
    // タイミング記録
    std::vector<RPPGTiming> timing_log;
    
    // 顔検出の最適化用変数
    std::vector<cv::Point2f> last_landmarks;
    int consecutive_no_face = 0;
    static const int MAX_NO_FACE_FRAMES = 10;
    
    // メモリ再利用用の変数（最適化追加）
    cv::Mat reusable_blob;
    cv::Mat reusable_face_roi;
    cv::Mat reusable_mask;
    cv::Mat reusable_ycbcr;
    cv::Mat reusable_skin_mask;
    cv::Mat reusable_combined_mask;
    
    // バッチ処理用の変数
    std::vector<cv::Mat> frame_batch;
    static const int BATCH_SIZE_FACE_DETECTION = 4;
    
    Impl(const std::string& model_directory = "");
    
    // タイミング計測ヘルパー関数
    void start_timing(const std::string& stage_name);
    void end_timing();
    void end_current_timing();
    void clear_timing_log();
    std::string get_timing_summary() const;
    
    // 顔検出関数
    std::vector<cv::Point2f> detectFaceLandmarks(const cv::Mat& frame);
    
    // 最適化されたROI処理関数
    cv::Scalar processROI(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks);
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
    std::string ffmpeg_cmd = ffmpeg_bin + " -y -fflags +genpts -err_detect ignore_err -i \"" + videoPath + "\" -q:v 2 \"" + temp_dir + "/frame_%05d.jpg\" > \"" + ffmpeg_log + "\" 2>&1";
    // std::string ffmpeg_cmd = ffmpeg_bin + " -y -fflags +genpts -err_detect ignore_err -i \"" + videoPath + "\" -vf \"scale=320:240\" -q:v 2 \"" + temp_dir + "/frame_%05d.jpg\" > \"" + ffmpeg_log + "\" 2>&1";
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
    
    // バッチサイズを設定（メモリ効率のため）
    const int BATCH_SIZE = 100;
    const int total_frames = static_cast<int>(imagePaths.size());
    
    // 進捗表示
    std::cout << "[RPPG] Processing " << total_frames << " frames in batches of " << BATCH_SIZE << std::endl;
    
    for (int batch_start = 0; batch_start < total_frames; batch_start += BATCH_SIZE) {
        int batch_end = std::min(batch_start + BATCH_SIZE, total_frames);
        
        // バッチ内のフレームを処理
        for (int i = batch_start; i < batch_end; ++i) {
            const auto& path = imagePaths[i];
            
            // 画像読み込みを最適化
            cv::Mat frame = cv::imread(path, cv::IMREAD_COLOR);
            if (frame.empty()) {
                std::cerr << "[RPPG] Warning: Could not read image: " << path << std::endl;
                continue;
            }
            
            double current_time = frame_count / fps;
            frame_count++;
            
            // 顔検出とROI抽出（最適化版）
            std::vector<cv::Point2f> landmarks;
            try {
                landmarks = pImpl->detectFaceLandmarks(frame);
            } catch (const cv::Exception& e) {
                std::cerr << "[RPPG] Face detection error: " << e.what() << std::endl;
                continue;
            }
            
            if (!landmarks.empty()) {
                // 最適化されたROI処理
                cv::Scalar mean = pImpl->processROI(frame, landmarks);
                
                // 有効な平均値のみを保存
                if (mean[0] > 0 && mean[1] > 0 && mean[2] > 0) {
                    skin_means.push_back({mean[0], mean[1], mean[2]});
                    timestamps.push_back(current_time);
                }
            }
        }
        
        // バッチ処理後の進捗表示
        if (batch_end % (BATCH_SIZE * 2) == 0 || batch_end == total_frames) {
            std::cout << "[RPPG] Processed " << batch_end << "/" << total_frames << " frames..." << std::endl;
        }
    }
    pImpl->end_timing(); // Image Loading and Processing
    
    std::cout << "[RPPG] Total processed frames: " << frame_count << ", Valid skin means: " << skin_means.size() << std::endl;
    
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
    
    // すべてのアクティブなタイミングを終了
    pImpl->end_current_timing();
    
    RPPGResult result;
    result.rppg_signal = rppg_signal;
    result.time_data = time_data;
    result.peak_times = peak_times;
    return result;
} 

// Impl構造体のメンバー関数実装
RPPGProcessor::Impl::Impl(const std::string& model_directory) : model_dir(model_directory) {
    // Initialize OpenCV DNN face detector with optimization
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
        
        // OpenCV最適化設定
        face_detector.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        face_detector.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        dnn_initialized = true;
        std::cout << "[RPPG] OpenCV DNN face detector initialized successfully with optimizations" << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "Failed to load OpenCV DNN face detector: " << e.what() << std::endl;
        dnn_initialized = false;
    }
}

void RPPGProcessor::Impl::start_timing(const std::string& stage_name) {
    RPPGTiming timing;
    timing.start(stage_name);
    timing_log.push_back(timing);
}

void RPPGProcessor::Impl::end_timing() {
    if (!timing_log.empty()) {
        timing_log.back().end();
    }
}

void RPPGProcessor::Impl::end_current_timing() {
    for (auto& timing : timing_log) {
        if (timing.is_active) {
            timing.end();
        }
    }
}

void RPPGProcessor::Impl::clear_timing_log() {
    timing_log.clear();
}

std::string RPPGProcessor::Impl::get_timing_summary() const {
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

std::vector<cv::Point2f> RPPGProcessor::Impl::detectFaceLandmarks(const cv::Mat& frame) {
    std::vector<cv::Point2f> landmarks;
    
    if (!dnn_initialized) {
        std::cerr << "[RPPG] Face detector not initialized" << std::endl;
        return landmarks;
    }
    
    // フレームサイズが小さすぎる場合はスキップ
    if (frame.cols < 100 || frame.rows < 100) {
        return landmarks;
    }
    
    // 前回の顔位置を再利用（連続で顔が見つからない場合）
    if (consecutive_no_face < MAX_NO_FACE_FRAMES && !last_landmarks.empty()) {
        landmarks = last_landmarks;
        consecutive_no_face++;
        return landmarks;
    }
    
    // 最適化されたblob作成（メモリ再利用）
    if (reusable_blob.empty() || reusable_blob.size() != cv::Size(300, 300)) {
        reusable_blob = cv::Mat(1, 3, 300, 300, CV_32F);
    }
    
    // 高速なblob作成
    cv::dnn::blobFromImage(frame, reusable_blob, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0), false, false);
    face_detector.setInput(reusable_blob);
    
    // Forward pass
    cv::Mat detections = face_detector.forward();
    
    // Process detections
    cv::Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
    
    float max_confidence = 0.0;
    std::vector<cv::Point2f> best_landmarks;
    
    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);
        
        if (confidence > 0.5 && confidence > max_confidence) { // Confidence threshold
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
            
            // 顔のサイズが適切かチェック
            int face_width = x2 - x1;
            int face_height = y2 - y1;
            if (face_width < 50 || face_height < 50) continue; // 小さすぎる顔はスキップ
            
            // Create face ROI landmarks (rectangular)
            best_landmarks.clear();
            best_landmarks.push_back(cv::Point2f(x1, y1));
            best_landmarks.push_back(cv::Point2f(x2, y1));
            best_landmarks.push_back(cv::Point2f(x2, y2));
            best_landmarks.push_back(cv::Point2f(x1, y2));
            
            // Add more points for better ROI coverage
            best_landmarks.push_back(cv::Point2f((x1 + x2) / 2, y1));
            best_landmarks.push_back(cv::Point2f((x1 + x2) / 2, y2));
            best_landmarks.push_back(cv::Point2f(x1, (y1 + y2) / 2));
            best_landmarks.push_back(cv::Point2f(x2, (y1 + y2) / 2));
            
            max_confidence = confidence;
        }
    }
    
    // 顔が見つかった場合
    if (!best_landmarks.empty()) {
        last_landmarks = best_landmarks;
        consecutive_no_face = 0;
        landmarks = best_landmarks;
    } else {
        consecutive_no_face++;
    }
    
    return landmarks;
}

// 最適化されたcv::Mat使用例
cv::Scalar RPPGProcessor::Impl::processROI_optimized(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.empty()) {
        return cv::Scalar(0, 0, 0);
    }
    
    // 1. ビューを使用（深いコピーを避ける）
    cv::Rect roi_rect = calculateROIRect(frame, landmarks);
    cv::Mat roi_view = frame(roi_rect);  // ビュー（コピーなし）
    
    // 2. メモリ再利用（事前割り当て）
    if (reusable_mask.empty() || reusable_mask.size() != roi_view.size()) {
        reusable_mask = cv::Mat::zeros(roi_view.size(), CV_8UC1);
    } else {
        reusable_mask.setTo(0);  // 既存メモリを再利用
    }
    
    // 3. 効率的なマスク作成
    createROIMask(landmarks, roi_rect, reusable_mask);
    
    // 4. 色空間変換（メモリ再利用）
    if (reusable_ycbcr.empty() || reusable_ycbcr.size() != roi_view.size()) {
        reusable_ycbcr = cv::Mat(roi_view.size(), CV_8UC3);
    }
    cv::cvtColor(roi_view, reusable_ycbcr, cv::COLOR_BGR2YCrCb);
    
    // 5. 肌色フィルタリング（メモリ再利用）
    if (reusable_skin_mask.empty() || reusable_skin_mask.size() != roi_view.size()) {
        reusable_skin_mask = cv::Mat(roi_view.size(), CV_8UC1);
    }
    cv::inRange(reusable_ycbcr, cv::Scalar(0, 100, 130), cv::Scalar(255, 140, 175), reusable_skin_mask);
    
    // 6. マスク組み合わせ（メモリ再利用）
    if (reusable_combined_mask.empty() || reusable_combined_mask.size() != roi_view.size()) {
        reusable_combined_mask = cv::Mat(roi_view.size(), CV_8UC1);
    }
    cv::bitwise_and(reusable_mask, reusable_skin_mask, reusable_combined_mask);
    
    // 7. 平均計算
    return cv::mean(roi_view, reusable_combined_mask);
}

// ROI矩形計算（効率的）
cv::Rect RPPGProcessor::Impl::calculateROIRect(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks) {
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    
    for (const auto& pt : landmarks) {
        min_x = std::min(min_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_x = std::max(max_x, pt.x);
        max_y = std::max(max_y, pt.y);
    }
    
    float margin = 20.0f;
    int x1 = std::max(0, static_cast<int>(min_x - margin));
    int y1 = std::max(0, static_cast<int>(min_y - margin));
    int x2 = std::min(frame.cols - 1, static_cast<int>(max_x + margin));
    int y2 = std::min(frame.rows - 1, static_cast<int>(max_y + margin));
    
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

// 効率的なマスク作成
void RPPGProcessor::Impl::createROIMask(const std::vector<cv::Point2f>& landmarks, 
                                       const cv::Rect& roi_rect, cv::Mat& mask) {
    std::vector<cv::Point> roi_points;
    for (const auto& pt : landmarks) {
        int rel_x = static_cast<int>(pt.x - roi_rect.x);
        int rel_y = static_cast<int>(pt.y - roi_rect.y);
        roi_points.push_back(cv::Point(rel_x, rel_y));
    }
    
    if (roi_points.size() >= 3) {
        cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{roi_points}, cv::Scalar(255));
    }
} 

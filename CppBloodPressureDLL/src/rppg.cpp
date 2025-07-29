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
    bool is_active;
    
    RPPGTiming() : is_active(false) {}
    
    void start(const std::string& name) {
        stage_name = name;
        start_time = std::chrono::high_resolution_clock::now();
        is_active = true;
    }
    
    void end() {
        if (is_active) {
            end_time = std::chrono::high_resolution_clock::now();
            is_active = false;
        }
    }
    
    double get_duration_ms() const {
        if (!is_active) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            return duration.count() / 1000.0;
        }
        return 0.0;
    }
};

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
    
    // 現在アクティブなタイミングを終了
    void end_current_timing() {
        for (auto& timing : timing_log) {
            if (timing.is_active) {
                timing.end();
            }
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
        
        // Prepare input blob with optimized size
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0), false, false);
        face_detector.setInput(blob);
        
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
            
            // 顔検出とROI抽出
            std::vector<cv::Point2f> landmarks;
            try {
                landmarks = pImpl->detectFaceLandmarks(frame);
            } catch (const cv::Exception& e) {
                std::cerr << "[RPPG] Face detection error: " << e.what() << std::endl;
                continue;
            }
            
            if (!landmarks.empty()) {
                // ROIマスクの作成
                cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
                std::vector<cv::Point> roi_points;
                for (const auto& pt : landmarks) {
                    roi_points.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
                }
                if (roi_points.size() >= 3) {
                    try {
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
                    } catch (const cv::Exception& e) {
                        std::cerr << "[RPPG] Image processing error: " << e.what() << std::endl;
                        continue;
                    }
                }
            }
        }
        
        // バッチ処理後の進捗表示
        if (batch_end % (BATCH_SIZE * 2) == 0 || batch_end == total_frames) {
            std::cout << "[RPPG] Processed " << batch_end << "/" << total_frames << " frames..." << std::endl;
        }
        
        // メモリクリーンアップは一時的に無効化（OpenCVエラー回避のため）
        // if (batch_end % (BATCH_SIZE * 3) == 0) {
        //     // 安全なメモリクリーンアップ
        //     try {
        //         // 明示的にメモリを解放（より安全な方法）
        //         cv::Mat temp;
        //         temp = cv::Mat();
        //     } catch (const cv::Exception& e) {
        //         std::cerr << "[RPPG] Memory cleanup warning: " << e.what() << std::endl;
        //     }
        // }
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

RPPGResult RPPGProcessor::processVideoDirect(const std::string& videoPath) {
    // タイミングログをクリア
    pImpl->clear_timing_log();
    
    pImpl->start_timing("RPPG Total Processing");
    
    std::vector<std::vector<double>> skin_means;
    std::vector<double> timestamps;
    int frame_count = 0;
    
    pImpl->start_timing("Video Loading and Processing");
    
    // OpenCVで動画を直接読み込み
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[RPPG] Error: Could not open video file: " << videoPath << std::endl;
        return RPPGResult();
    }
    
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "[RPPG] Video info - FPS: " << fps << ", Total frames: " << total_frames << std::endl;
    
    cv::Mat frame;
    while (cap.read(frame)) {
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
        
        // 進捗表示（100フレームごと）
        if (frame_count % 100 == 0) {
            std::cout << "[RPPG] Processed " << frame_count << " frames..." << std::endl;
        }
    }
    
    cap.release();
    pImpl->end_timing(); // Video Loading and Processing
    
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

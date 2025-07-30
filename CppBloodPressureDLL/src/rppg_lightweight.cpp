#include "rppg.h"
#include "peak_detect.h"
#include <iostream>
// 必要最小限のOpenCVヘッダーのみ使用（軽量化）
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>

// dlibを使用した高精度顔検出・ランドマーク検出
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

// 軽量版RPPGProcessor（OpenCV DNN不使用）
class LightweightRPPGProcessor {
private:
    // dlib顔検出器
    dlib::frontal_face_detector detector_;
    dlib::shape_predictor predictor_;
    std::vector<cv::Point2f> last_landmarks_;
    int consecutive_no_face_ = 0;
    static const int MAX_NO_FACE_FRAMES = 10;
    
    // メモリ再利用用の変数
    cv::Mat reusable_gray;
    cv::Mat reusable_ycbcr;
    cv::Mat reusable_skin_mask;
    cv::Mat reusable_combined_mask;
    
    // タイミング記録
    std::vector<RPPGTiming> timing_log;
    
public:
    bool Initialize(const std::string& model_dir = "") {
        try {
            // dlib顔検出器初期化
            detector_ = dlib::get_frontal_face_detector();
            
            // ランドマーク予測器の読み込み
            std::vector<std::string> predictor_paths = {
                "shape_predictor_68_face_landmarks.dat",
                model_dir + "/shape_predictor_68_face_landmarks.dat",
                "models/shape_predictor_68_face_landmarks.dat"
            };
            
            bool loaded = false;
            for (const auto& path : predictor_paths) {
                try {
                    dlib::deserialize(path) >> predictor_;
                    std::cout << "[RPPG] dlib shape predictor loaded: " << path << std::endl;
                    loaded = true;
                    break;
                } catch (const dlib::serialization_error& e) {
                    std::cerr << "[RPPG] Failed to load dlib predictor from: " << path << std::endl;
                }
            }
            
            if (!loaded) {
                std::cerr << "[RPPG] Failed to load dlib shape predictor" << std::endl;
                return false;
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[RPPG] dlib initialization error: " << e.what() << std::endl;
            return false;
        }
    }
    
    std::vector<cv::Point2f> DetectLandmarks(const cv::Mat& frame) {
        // 前回の結果を再利用（安定性向上）
        if (consecutive_no_face_ < MAX_NO_FACE_FRAMES && !last_landmarks_.empty()) {
            consecutive_no_face_++;
            return last_landmarks_;
        }
        
        try {
            // OpenCV Matをdlib形式に変換
            dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);
            
            // 顔検出
            std::vector<dlib::rectangle> faces = detector_(dlib_img);
            
            if (!faces.empty()) {
                // 最大の顔を選択
                dlib::rectangle best_face = faces[0];
                for (const auto& face : faces) {
                    if (face.area() > best_face.area()) {
                        best_face = face;
                    }
                }
                
                // 68個のランドマークを取得
                dlib::full_object_detection shape = predictor_(dlib_img, best_face);
                
                // rPPG用途に最適なランドマークを選択（頬、額、鼻周辺）
                std::vector<cv::Point2f> landmarks;
                
                // 主要なランドマークインデックス（rPPG用途）
                std::vector<int> roi_landmarks = {
                    // 左頬
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    // 右頬
                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                    // 額
                    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                    // 鼻
                    27, 28, 29, 30, 31, 32, 33, 34, 35,
                    // 口周辺
                    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67
                };
                
                for (int idx : roi_landmarks) {
                    if (idx < shape.num_parts()) {
                        auto part = shape.part(idx);
                        landmarks.push_back(cv::Point2f(part.x(), part.y()));
                    }
                }
                
                // 重複除去とソート
                std::sort(landmarks.begin(), landmarks.end(), 
                    [](const cv::Point2f& a, const cv::Point2f& b) {
                        return a.x < b.x || (a.x == b.x && a.y < b.y);
                    });
                landmarks.erase(std::unique(landmarks.begin(), landmarks.end()), landmarks.end());
                
                last_landmarks_ = landmarks;
                consecutive_no_face_ = 0;
                return landmarks;
            } else {
                consecutive_no_face_++;
                return last_landmarks_;
            }
        } catch (const std::exception& e) {
            std::cerr << "[RPPG] dlib detection error: " << e.what() << std::endl;
            consecutive_no_face_++;
            return last_landmarks_;
        }
    }
    
    cv::Scalar ProcessROI(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks) {
        if (landmarks.empty()) {
            return cv::Scalar(0, 0, 0);
        }
        
        // ROI矩形計算
        cv::Rect roi_rect = CalculateROIRect(frame, landmarks);
        
        // ROI抽出（ビュー使用でコピー回避）
        cv::Mat roi_view = frame(roi_rect);
        
        // 色空間変換（メモリ再利用）
        if (reusable_ycbcr.empty() || reusable_ycbcr.size() != roi_view.size()) {
            reusable_ycbcr = cv::Mat(roi_view.size(), CV_8UC3);
        }
        cv::cvtColor(roi_view, reusable_ycbcr, cv::COLOR_BGR2YCrCb);
        
        // 肌色フィルタリング（メモリ再利用）
        if (reusable_skin_mask.empty() || reusable_skin_mask.size() != roi_view.size()) {
            reusable_skin_mask = cv::Mat(roi_view.size(), CV_8UC1);
        }
        cv::inRange(reusable_ycbcr, cv::Scalar(0, 100, 130), cv::Scalar(255, 140, 175), reusable_skin_mask);
        
        // マスク組み合わせ（メモリ再利用）
        if (reusable_combined_mask.empty() || reusable_combined_mask.size() != roi_view.size()) {
            reusable_combined_mask = cv::Mat(roi_view.size(), CV_8UC1);
        }
        
        // 効率的なマスク作成
        CreateOptimizedMask(landmarks, roi_rect, reusable_combined_mask);
        cv::bitwise_and(reusable_combined_mask, reusable_skin_mask, reusable_combined_mask);
        
        // 平均計算
        return cv::mean(roi_view, reusable_combined_mask);
    }
    
private:
    cv::Rect CalculateROIRect(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks) {
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
        
        float margin = 15.0f;
        int x1 = std::max(0, static_cast<int>(min_x - margin));
        int y1 = std::max(0, static_cast<int>(min_y - margin));
        int x2 = std::min(frame.cols - 1, static_cast<int>(max_x + margin));
        int y2 = std::min(frame.rows - 1, static_cast<int>(max_y + margin));
        
        return cv::Rect(x1, y1, x2 - x1, y2 - y1);
    }
    
    void CreateOptimizedMask(const std::vector<cv::Point2f>& landmarks, 
                           const cv::Rect& roi_rect, cv::Mat& mask) {
        std::vector<cv::Point> roi_points;
        roi_points.reserve(landmarks.size());
        
        for (const auto& pt : landmarks) {
            int rel_x = static_cast<int>(pt.x - roi_rect.x);
            int rel_y = static_cast<int>(pt.y - roi_rect.y);
            roi_points.push_back(cv::Point(rel_x, rel_y));
        }
        
        if (roi_points.size() >= 3) {
            cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{roi_points}, cv::Scalar(255));
        }
    }
};

// 軽量版RPPGProcessorの実装
RPPGResult ProcessImagesLightweight(const std::vector<std::string>& imagePaths, double fps) {
    LightweightRPPGProcessor processor;
    if (!processor.Initialize()) {
        throw std::runtime_error("Failed to initialize lightweight RPPG processor");
    }
    
    std::vector<std::vector<double>> skin_means;
    std::vector<double> timestamps;
    int frame_count = 0;
    
    std::cout << "[RPPG] Processing " << imagePaths.size() << " frames with lightweight version..." << std::endl;
    
    for (const auto& path : imagePaths) {
        cv::Mat frame = cv::imread(path, cv::IMREAD_COLOR);
        if (frame.empty()) {
            std::cerr << "[RPPG] Warning: Could not read image: " << path << std::endl;
            continue;
        }
        
        double current_time = frame_count / fps;
        frame_count++;
        
        // 顔検出とROI抽出
        std::vector<cv::Point2f> landmarks = processor.DetectLandmarks(frame);
        
        if (!landmarks.empty()) {
            cv::Scalar mean = processor.ProcessROI(frame, landmarks);
            
            if (mean[0] > 0 && mean[1] > 0 && mean[2] > 0) {
                skin_means.push_back({mean[0], mean[1], mean[2]});
                timestamps.push_back(current_time);
            }
        }
    }
    
    std::cout << "[RPPG] Total processed frames: " << frame_count << ", Valid skin means: " << skin_means.size() << std::endl;
    
    // RPPG信号処理（既存の実装と同じ）
    std::vector<double> rppg_signal, time_data, peak_times;
    if (skin_means.size() > 30) {
        int L = 30;
        int frames = static_cast<int>(skin_means.size());
        Eigen::MatrixXd C(3, frames);
        for (int i = 0; i < frames; ++i) {
            C(0, i) = skin_means[i][0];
            C(1, i) = skin_means[i][1];
            C(2, i) = skin_means[i][2];
        }
        
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
        
        double mean = H.mean();
        double stddev = std::sqrt((H.array() - mean).square().sum() / H.size());
        Eigen::VectorXd pulse_z = (H.array() - mean) / (stddev + 1e-8);
        rppg_signal.assign(pulse_z.data(), pulse_z.data() + pulse_z.size());
        time_data = timestamps;
        
        std::vector<size_t> peaks = find_peaks(rppg_signal, 10, 0.0);
        for (size_t idx : peaks) {
            if (idx < time_data.size()) {
                peak_times.push_back(time_data[idx]);
            }
        }
    }
    
    RPPGResult result;
    result.rppg_signal = rppg_signal;
    result.time_data = time_data;
    result.peak_times = peak_times;
    return result;
} 

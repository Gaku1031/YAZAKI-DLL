#include "rppg.h"
#include "peak_detect.h"
#include <iostream>
// 必要最小限のOpenCVヘッダーのみ使用（軽量化）
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
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

// dlibを使用した高精度顔検出・ランドマーク検出
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

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

// Impl構造体の完全な定義
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
    cv::Scalar processROI_optimized(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks);
    
    // 効率的なヘルパー関数
    cv::Rect calculateROIRect(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks);
    void createROIMask(const std::vector<cv::Point2f>& landmarks, const cv::Rect& roi_rect, cv::Mat& mask);
};

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

// 軽量で高速な顔検出実装（OpenCV Haar Cascade）
class LightweightFaceDetector {
private:
    cv::CascadeClassifier face_cascade_;
    cv::CascadeClassifier eye_cascade_;
    std::vector<cv::Point2f> last_landmarks_;
    int consecutive_no_face_ = 0;
    static const int MAX_NO_FACE_FRAMES = 15;
    
public:
    bool Initialize(const std::string& model_dir = "") {
        std::vector<std::string> cascade_paths = {
            "haarcascade_frontalface_alt2.xml",
            model_dir + "/haarcascade_frontalface_alt2.xml",
            "models/haarcascade_frontalface_alt2.xml"
        };
        
        bool loaded = false;
        for (const auto& path : cascade_paths) {
            if (face_cascade_.load(path)) {
                std::cout << "[RPPG] Haar cascade loaded: " << path << std::endl;
                loaded = true;
                break;
            }
        }
        
        if (!loaded) {
            std::cerr << "[RPPG] Failed to load Haar cascade" << std::endl;
            return false;
        }
        
        return true;
    }
    
    std::vector<cv::Point2f> DetectLandmarks(const cv::Mat& frame) {
        // 前回の結果を再利用（安定性向上）
        if (consecutive_no_face_ < MAX_NO_FACE_FRAMES && !last_landmarks_.empty()) {
            consecutive_no_face_++;
            return last_landmarks_;
        }
        
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);  // コントラスト改善
        
        std::vector<cv::Rect> faces;
        face_cascade_.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(50, 50));
        
        if (!faces.empty()) {
            // 最大の顔を選択
            cv::Rect best_face = faces[0];
            for (const auto& face : faces) {
                if (face.area() > best_face.area()) {
                    best_face = face;
                }
            }
            
            // 8点のランドマークを生成
            std::vector<cv::Point2f> landmarks;
            landmarks.push_back(cv::Point2f(best_face.x, best_face.y));
            landmarks.push_back(cv::Point2f(best_face.x + best_face.width, best_face.y));
            landmarks.push_back(cv::Point2f(best_face.x + best_face.width, best_face.y + best_face.height));
            landmarks.push_back(cv::Point2f(best_face.x, best_face.y + best_face.height));
            
            // 中心点と中間点を追加
            landmarks.push_back(cv::Point2f(best_face.x + best_face.width/2, best_face.y));
            landmarks.push_back(cv::Point2f(best_face.x + best_face.width/2, best_face.y + best_face.height));
            landmarks.push_back(cv::Point2f(best_face.x, best_face.y + best_face.height/2));
            landmarks.push_back(cv::Point2f(best_face.x + best_face.width, best_face.y + best_face.height/2));
            
            last_landmarks_ = landmarks;
            consecutive_no_face_ = 0;
            return landmarks;
        } else {
            consecutive_no_face_++;
            return last_landmarks_;
        }
    }
};

// dlibを使用した高精度顔検出・ランドマーク検出
class DlibFaceDetector {
private:
    dlib::frontal_face_detector detector_;
    dlib::shape_predictor predictor_;
    std::vector<cv::Point2f> last_landmarks_;
    int consecutive_no_face_ = 0;
    static const int MAX_NO_FACE_FRAMES = 10;
    
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
    
    // rPPG用途に特化したランドマーク取得
    std::vector<cv::Point2f> GetRPPGLandmarks(const cv::Mat& frame) {
        auto all_landmarks = DetectLandmarks(frame);
        
        if (all_landmarks.empty()) {
            return {};
        }
        
        // rPPG用途に最適な領域を選択
        std::vector<cv::Point2f> rppg_landmarks;
        
        // 頬領域（血流検出に最適）
        for (size_t i = 0; i < all_landmarks.size(); ++i) {
            const auto& pt = all_landmarks[i];
            // 頬領域のランドマークを選択（簡易フィルタ）
            if (pt.y > frame.rows * 0.3 && pt.y < frame.rows * 0.7) {
                rppg_landmarks.push_back(pt);
            }
        }
        
        return rppg_landmarks;
    }
};

// 既存の実装をdlib版に置き換え
std::vector<cv::Point2f> RPPGProcessor::Impl::detectFaceLandmarks(const cv::Mat& frame) {
    static DlibFaceDetector detector;
    static bool initialized = false;
    
    if (!initialized) {
        initialized = detector.Initialize(model_dir);
        if (!initialized) {
            std::cerr << "[RPPG] dlib face detector initialization failed" << std::endl;
            return {};
        }
    }
    
    return detector.GetRPPGLandmarks(frame);
}

// dlib使用時のOpenCV画像処理最適化
class OptimizedImageProcessor {
private:
    // メモリ再利用用の変数
    cv::Mat reusable_gray;
    cv::Mat reusable_roi;
    cv::Mat reusable_ycbcr;
    cv::Mat reusable_skin_mask;
    cv::Mat reusable_combined_mask;
    
    // 前回のROI領域をキャッシュ
    cv::Rect last_roi_rect;
    bool roi_cache_valid = false;
    
public:
    // 最適化されたROI処理
    cv::Scalar ProcessROI_Optimized(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks) {
        if (landmarks.empty()) {
            return cv::Scalar(0, 0, 0);
        }
        
        // 1. ROI矩形計算（キャッシュ活用）
        cv::Rect roi_rect = CalculateROIRect(frame, landmarks);
        
        // 2. ROIキャッシュチェック（前回と同じ領域なら再利用）
        if (roi_cache_valid && roi_rect == last_roi_rect) {
            // キャッシュされたROIを使用
            return ProcessCachedROI(frame, roi_rect);
        }
        
        // 3. 新しいROI処理
        last_roi_rect = roi_rect;
        roi_cache_valid = true;
        
        // 4. グレースケール変換（メモリ再利用）
        if (reusable_gray.empty() || reusable_gray.size() != roi_rect.size()) {
            reusable_gray = cv::Mat(roi_rect.size(), CV_8UC1);
        }
        cv::cvtColor(frame(roi_rect), reusable_gray, cv::COLOR_BGR2GRAY);
        
        // 5. ROI抽出（ビュー使用でコピー回避）
        cv::Mat roi_view = frame(roi_rect);
        
        // 6. 色空間変換（メモリ再利用）
        if (reusable_ycbcr.empty() || reusable_ycbcr.size() != roi_view.size()) {
            reusable_ycbcr = cv::Mat(roi_view.size(), CV_8UC3);
        }
        cv::cvtColor(roi_view, reusable_ycbcr, cv::COLOR_BGR2YCrCb);
        
        // 7. 肌色フィルタリング（メモリ再利用）
        if (reusable_skin_mask.empty() || reusable_skin_mask.size() != roi_view.size()) {
            reusable_skin_mask = cv::Mat(roi_view.size(), CV_8UC1);
        }
        cv::inRange(reusable_ycbcr, cv::Scalar(0, 100, 130), cv::Scalar(255, 140, 175), reusable_skin_mask);
        
        // 8. マスク組み合わせ（メモリ再利用）
        if (reusable_combined_mask.empty() || reusable_combined_mask.size() != roi_view.size()) {
            reusable_combined_mask = cv::Mat(roi_view.size(), CV_8UC1);
        }
        
        // 9. 効率的なマスク作成
        CreateOptimizedMask(landmarks, roi_rect, reusable_combined_mask);
        cv::bitwise_and(reusable_combined_mask, reusable_skin_mask, reusable_combined_mask);
        
        // 10. 平均計算
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
        
        float margin = 15.0f;  // マージンを小さくして処理領域削減
        int x1 = std::max(0, static_cast<int>(min_x - margin));
        int y1 = std::max(0, static_cast<int>(min_y - margin));
        int x2 = std::min(frame.cols - 1, static_cast<int>(max_x + margin));
        int y2 = std::min(frame.rows - 1, static_cast<int>(max_y + margin));
        
        return cv::Rect(x1, y1, x2 - x1, y2 - y1);
    }
    
    void CreateOptimizedMask(const std::vector<cv::Point2f>& landmarks, 
                           const cv::Rect& roi_rect, cv::Mat& mask) {
        std::vector<cv::Point> roi_points;
        roi_points.reserve(landmarks.size());  // メモリ事前割り当て
        
        for (const auto& pt : landmarks) {
            int rel_x = static_cast<int>(pt.x - roi_rect.x);
            int rel_y = static_cast<int>(pt.y - roi_rect.y);
            roi_points.push_back(cv::Point(rel_x, rel_y));
        }
        
        if (roi_points.size() >= 3) {
            cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{roi_points}, cv::Scalar(255));
        }
    }
    
    cv::Scalar ProcessCachedROI(const cv::Mat& frame, const cv::Rect& roi_rect) {
        // キャッシュされたROIの処理（高速化）
        cv::Mat roi_view = frame(roi_rect);
        
        // 簡易処理（キャッシュ時は軽量版）
        cv::Mat gray_roi;
        cv::cvtColor(roi_view, gray_roi, cv::COLOR_BGR2GRAY);
        
        // 緑チャンネル強調（血流検出）
        cv::Mat green_channel;
        std::vector<cv::Mat> channels;
        cv::split(roi_view, channels);
        green_channel = channels[1];
        
        // 簡易平均計算
        cv::Scalar mean_green = cv::mean(green_channel);
        return cv::Scalar(mean_green[0], mean_green[0], mean_green[0]);
    }
};

// dlib使用時の最適化されたROI処理
cv::Scalar RPPGProcessor::Impl::processROI_optimized(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks) {
    static OptimizedImageProcessor processor;
    return processor.ProcessROI_Optimized(frame, landmarks);
}

// 既存のROI処理関数（後方互換性のため）
cv::Scalar RPPGProcessor::Impl::processROI(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks) {
    return processROI_optimized(frame, landmarks);
}

// ROI矩形計算ヘルパー関数
cv::Rect RPPGProcessor::Impl::calculateROIRect(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.empty()) {
        return cv::Rect(0, 0, frame.cols, frame.rows);
    }
    
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

// ROIマスク作成ヘルパー関数
void RPPGProcessor::Impl::createROIMask(const std::vector<cv::Point2f>& landmarks, 
                                       const cv::Rect& roi_rect, cv::Mat& mask) {
    if (landmarks.empty()) {
        mask.setTo(255);
        return;
    }
    
    std::vector<cv::Point> roi_points;
    roi_points.reserve(landmarks.size());
    
    for (const auto& pt : landmarks) {
        int rel_x = static_cast<int>(pt.x - roi_rect.x);
        int rel_y = static_cast<int>(pt.y - roi_rect.y);
        roi_points.push_back(cv::Point(rel_x, rel_y));
    }
    
    if (roi_points.size() >= 3) {
        cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{roi_points}, cv::Scalar(255));
    } else {
        mask.setTo(255);
    }
}

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

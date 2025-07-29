#pragma once
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <chrono>

// 詳細タイミング計測用の構造体
struct RPPGTiming {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    std::string stage_name;
    bool is_active;
    
    RPPGTiming();
    void start(const std::string& name);
    void end();
    double get_duration_ms() const;
};

struct RPPGResult {
    std::vector<double> rppg_signal;
    std::vector<double> time_data;
    std::vector<double> peak_times;
};

class RPPGProcessor {
public:
    RPPGProcessor(const std::string& model_dir = "");
    ~RPPGProcessor();
    RPPGResult processVideo(const std::string& videoPath);
    RPPGResult processImagesFromPaths(const std::vector<std::string>& imagePaths, double fps = 30.0);
    std::string get_timing_summary() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

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
    cv::Scalar processROI_optimized(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks);
    
    // 効率的なヘルパー関数
    cv::Rect calculateROIRect(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks);
    void createROIMask(const std::vector<cv::Point2f>& landmarks, const cv::Rect& roi_rect, cv::Mat& mask);
}; 

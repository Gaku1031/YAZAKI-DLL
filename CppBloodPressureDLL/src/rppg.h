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

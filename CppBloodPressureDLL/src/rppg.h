#pragma once
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

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
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
}; 

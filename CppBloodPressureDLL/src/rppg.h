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
    RPPGProcessor();
    ~RPPGProcessor();
    RPPGResult processVideo(const std::string& videoPath);
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
}; 

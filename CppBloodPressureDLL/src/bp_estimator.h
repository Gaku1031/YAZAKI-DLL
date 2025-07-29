#pragma once
#include <vector>
#include <string>
#include <memory>

class BloodPressureEstimator {
public:
    BloodPressureEstimator(const std::string& model_dir);
    ~BloodPressureEstimator();
    std::pair<int, int> estimate_bp(const std::vector<double>& peak_times, int height, int weight, int sex);
    std::string get_model_dir() const;
    std::string get_timing_summary() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    std::string model_directory;
}; 

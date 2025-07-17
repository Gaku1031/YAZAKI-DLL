#pragma once
#include <vector>
#include <string>
#include <memory>

class BloodPressureEstimator {
public:
    BloodPressureEstimator(const std::string& model_dir);
    ~BloodPressureEstimator();
    std::pair<int, int> estimate_bp(const std::vector<double>& peak_times, int height, int weight, int sex);
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
}; 

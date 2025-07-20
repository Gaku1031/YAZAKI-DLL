#include "bp_estimator.h"
#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class BPOpenCVImpl {
public:
    cv::dnn::Net sbp_net;
    cv::dnn::Net dbp_net;
    bool models_loaded;

    BPOpenCVImpl(const std::string& model_dir)
        : models_loaded(false)
    {
        try {
            std::string sbp_path = model_dir + "/model_sbp.onnx";
            std::string dbp_path = model_dir + "/model_dbp.onnx";
            
            sbp_net = cv::dnn::readNetFromONNX(sbp_path);
            dbp_net = cv::dnn::readNetFromONNX(dbp_path);
            
            models_loaded = true;
        } catch (const cv::Exception& e) {
            throw std::runtime_error("Failed to load ONNX models with OpenCV DNN: " + std::string(e.what()));
        }
    }

    float run(const cv::dnn::Net& net, const std::vector<float>& features) {
        if (!models_loaded) {
            throw std::runtime_error("Models not loaded");
        }

        // Create input blob
        cv::Mat input_blob = cv::Mat(1, features.size(), CV_32F);
        for (size_t i = 0; i < features.size(); ++i) {
            input_blob.at<float>(0, i) = features[i];
        }

        // Set input and run inference
        net.setInput(input_blob);
        cv::Mat output = net.forward();

        // Get output value
        return output.at<float>(0, 0);
    }
};

struct BloodPressureEstimator::Impl {
    std::unique_ptr<BPOpenCVImpl> opencv_impl;
};

BloodPressureEstimator::BloodPressureEstimator(const std::string& model_dir)
    : pImpl(new Impl)
{
    pImpl->opencv_impl = std::make_unique<BPOpenCVImpl>(model_dir);
}

BloodPressureEstimator::~BloodPressureEstimator() = default;

std::pair<int, int> BloodPressureEstimator::estimate_bp(const std::vector<double>& peak_times, int height, int weight, int sex) {
    std::vector<double> rri;
    for (size_t i = 1; i < peak_times.size(); ++i) {
        double r = peak_times[i] - peak_times[i - 1];
        if (r > 0.4 && r < 1.2) rri.push_back(r);
    }
    if (rri.empty()) throw std::runtime_error("有効なRRIが検出されません");
    
    double mean = std::accumulate(rri.begin(), rri.end(), 0.0) / rri.size();
    double sq_sum = std::inner_product(rri.begin(), rri.end(), rri.begin(), 0.0);
    double stddev = std::sqrt(sq_sum / rri.size() - mean * mean);
    double min = *std::min_element(rri.begin(), rri.end());
    double max = *std::max_element(rri.begin(), rri.end());
    double bmi = weight / ((height / 100.0) * (height / 100.0));
    int sex_feature = (sex == 1) ? 1 : 0;
    std::vector<float> features = {float(mean), float(stddev), float(min), float(max), float(bmi), float(sex_feature)};

    float sbp = pImpl->opencv_impl->run(pImpl->opencv_impl->sbp_net, features);
    float dbp = pImpl->opencv_impl->run(pImpl->opencv_impl->dbp_net, features);

    return {int(std::round(sbp)), int(std::round(dbp))};
} 

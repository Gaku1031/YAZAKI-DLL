#include "bp_estimator.h"
#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#ifdef ONNXRUNTIME_AVAILABLE
#include <onnxruntime_cxx_api.h>
#endif

// OpenCV DNN fallback implementation
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

#ifdef ONNXRUNTIME_AVAILABLE
class BPONNXImpl {
public:
    Ort::Env env;
    Ort::Session sbp_session;
    Ort::Session dbp_session;
    Ort::SessionOptions session_options;

    BPONNXImpl(const std::string& model_dir)
        : env(ORT_LOGGING_LEVEL_WARNING, "bp"),
          sbp_session(nullptr), dbp_session(nullptr)
    {
        session_options.SetIntraOpNumThreads(1);
        std::string sbp_path = model_dir + "/model_sbp.onnx";
        std::string dbp_path = model_dir + "/model_dbp.onnx";
        sbp_session = Ort::Session(env, sbp_path.c_str(), session_options);
        dbp_session = Ort::Session(env, dbp_path.c_str(), session_options);
    }

    float run(const Ort::Session& session, const std::vector<float>& features) {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<int64_t> input_shape = {1, (int64_t)features.size()};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(features.data()), features.size(), input_shape.data(), input_shape.size());

        const char* input_name = session.GetInputName(0, allocator);
        const char* output_name = session.GetOutputName(0, allocator);
        std::vector<const char*> input_names = {input_name};
        std::vector<const char*> output_names = {output_name};

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

        float* out = output_tensors.front().GetTensorMutableData<float>();
        float result = out[0];

        allocator.Free((void*)input_name);
        allocator.Free((void*)output_name);

        return result;
    }
};
#else
class BPONNXImpl {
public:
    // Dummy session type for when ONNX Runtime is not available
    struct DummySession {};
    
    BPONNXImpl(const std::string& model_dir) {
        throw std::runtime_error("ONNX Runtime is required for blood pressure estimation");
    }

    float run(const DummySession& session, const std::vector<float>& features) {
        throw std::runtime_error("ONNX Runtime is required for blood pressure estimation");
    }
};
#endif

struct BloodPressureEstimator::Impl {
#ifdef ONNXRUNTIME_AVAILABLE
    std::unique_ptr<BPONNXImpl> onnx;
#else
    std::unique_ptr<BPOpenCVImpl> opencv;
#endif
};

BloodPressureEstimator::BloodPressureEstimator(const std::string& model_dir)
    : pImpl(new Impl)
{
#ifdef ONNXRUNTIME_AVAILABLE
    pImpl->onnx = std::make_unique<BPONNXImpl>(model_dir);
#else
    pImpl->opencv = std::make_unique<BPOpenCVImpl>(model_dir);
#endif
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

#ifdef ONNXRUNTIME_AVAILABLE
    float sbp = pImpl->onnx->run(pImpl->onnx->sbp_session, features);
    float dbp = pImpl->onnx->run(pImpl->onnx->dbp_session, features);
#else
    float sbp = pImpl->opencv->run(pImpl->opencv->sbp_net, features);
    float dbp = pImpl->opencv->run(pImpl->opencv->dbp_net, features);
#endif

    return {int(std::round(sbp)), int(std::round(dbp))};
} 

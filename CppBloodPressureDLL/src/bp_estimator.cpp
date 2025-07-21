#include "bp_estimator.h"
#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <memory>
#include <fstream>
#include <onnxruntime_cxx_api.h>

struct BloodPressureEstimator::Impl {
    Ort::Env env;
    Ort::Session sbp_session;
    Ort::Session dbp_session;
    Ort::SessionOptions session_options;

    Impl(const std::string& model_dir)
        : env(ORT_LOGGING_LEVEL_WARNING, "bp"),
          sbp_session(nullptr), dbp_session(nullptr)
    {
        session_options.SetIntraOpNumThreads(1);
        std::string sbp_path = model_dir + "/systolicbloodpressure.onnx";
        std::string dbp_path = model_dir + "/diastolicbloodpressure.onnx";
        // ファイル存在チェック
        std::ifstream sbp_file(sbp_path, std::ios::binary);
        std::ifstream dbp_file(dbp_path, std::ios::binary);
        if (!sbp_file.good()) {
            throw std::runtime_error("Systolic blood pressure model file not found: " + sbp_path);
        }
        if (!dbp_file.good()) {
            throw std::runtime_error("Diastolic blood pressure model file not found: " + dbp_path);
        }
        sbp_file.close();
        dbp_file.close();
        // Windows用にパスをwstringに変換
        std::wstring sbp_path_w(sbp_path.begin(), sbp_path.end());
        std::wstring dbp_path_w(dbp_path.begin(), dbp_path.end());
        sbp_session = Ort::Session(env, sbp_path_w.c_str(), session_options);
        dbp_session = Ort::Session(env, dbp_path_w.c_str(), session_options);
    }

    float run(Ort::Session& session, const std::vector<float>& features) {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<int64_t> input_shape = {1, (int64_t)features.size()};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(features.data()), features.size(), input_shape.data(), input_shape.size());
    
        // 入力・出力名の取得（Session経由）
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        std::vector<const char*> input_names = {input_name.get()};
        std::vector<const char*> output_names = {output_name.get()};
    
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
    
        float* out = output_tensors.front().GetTensorMutableData<float>();
        float result = out[0];
        return result;
    }
};

BloodPressureEstimator::BloodPressureEstimator(const std::string& model_dir)
    : pImpl(new Impl(model_dir))
{
    // ファイル存在チェック等はImplのコンストラクタで行う
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

    float sbp = pImpl->run(pImpl->sbp_session, features);
    float dbp = pImpl->run(pImpl->dbp_session, features);

    return {int(std::round(sbp)), int(std::round(dbp))};
} 

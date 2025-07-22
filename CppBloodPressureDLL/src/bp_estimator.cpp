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
    Ort::SessionOptions session_options;
    Ort::Session sbp_session;
    Ort::Session dbp_session;

    Impl(const std::string& model_dir)
        : env(ORT_LOGGING_LEVEL_WARNING, "bp"),
          session_options(),
          sbp_session(nullptr),
          dbp_session(nullptr)
    {
        try {
            std::string sbp_path = model_dir + "/systolicbloodpressure.onnx";
            std::string dbp_path = model_dir + "/diastolicbloodpressure.onnx";
            printf("[BP_EST] Loading models from: %s\n", model_dir.c_str()); fflush(stdout);
            printf("[BP_EST] SBP model path: %s\n", sbp_path.c_str()); fflush(stdout);
            printf("[BP_EST] DBP model path: %s\n", dbp_path.c_str()); fflush(stdout);

            // ファイル存在チェック
            std::ifstream sbp_file(sbp_path, std::ios::binary);
            std::ifstream dbp_file(dbp_path, std::ios::binary);
            if (!sbp_file.good()) throw std::runtime_error("SBP model file not found");
            if (!dbp_file.good()) throw std::runtime_error("DBP model file not found");

            // ファイルサイズチェック
            sbp_file.seekg(0, std::ios::end);
            auto sbp_size = sbp_file.tellg();
            dbp_file.seekg(0, std::ios::end);
            auto dbp_size = dbp_file.tellg();
            printf("[BP_EST] SBP model size: %ld bytes\n", (long)sbp_size); fflush(stdout);
            printf("[BP_EST] DBP model size: %ld bytes\n", (long)dbp_size); fflush(stdout);
            if (sbp_size == 0) throw std::runtime_error("SBP model file is empty: " + sbp_path);
            if (dbp_size == 0) throw std::runtime_error("DBP model file is empty: " + dbp_path);

            sbp_file.close();
            dbp_file.close();

            session_options.SetIntraOpNumThreads(1);
            printf("[BP_EST] Ort::SessionOptions created and threads set.\n"); fflush(stdout);
            std::wstring sbp_path_w(sbp_path.begin(), sbp_path.end());
            std::wstring dbp_path_w(dbp_path.begin(), dbp_path.end());
            sbp_session = Ort::Session(env, sbp_path_w.c_str(), session_options);
            printf("[BP_EST] SBP session created.\n"); fflush(stdout);
            dbp_session = Ort::Session(env, dbp_path_w.c_str(), session_options);
            printf("[BP_EST] DBP session created.\n"); fflush(stdout);
        } catch (const std::exception& e) {
            printf("[BP_EST] Standard exception in constructor: %s\n", e.what()); fflush(stdout);
            throw;
        }
    }

    float run(void* session_ptr, const std::vector<float>& input) {
        // session_ptrがnullptrならSBP、そうでなければDBPを使う（例）
        Ort::Session* session = session_ptr ? static_cast<Ort::Session*>(session_ptr) : &sbp_session;
        // 入力名・出力名はモデルに合わせて要調整
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};
        std::vector<int64_t> input_shape = {static_cast<int64_t>(input.size())};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input.data()), input.size(), input_shape.data(), input_shape.size());
        std::vector<Ort::Value> ort_outputs = session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        float result = 0.0f;
        if (ort_outputs.size() > 0 && ort_outputs[0].IsTensor()) {
            float* out_data = ort_outputs[0].GetTensorMutableData<float>();
            result = out_data[0];
        }
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
    // 入力ベクトルをfloatに変換
    std::vector<float> input;
    for (double v : peak_times) input.push_back(static_cast<float>(v));
    // SBP推定
    int sbp = static_cast<int>(std::round(pImpl->run(nullptr, input)));
    // DBP推定（例：session_ptrに&dbp_sessionを渡す）
    int dbp = static_cast<int>(std::round(pImpl->run(&pImpl->dbp_session, input)));
    return {sbp, dbp};
} 

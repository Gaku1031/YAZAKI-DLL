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

    float run(void*, const std::vector<float>&) {
        return 0.0f;
    }
};

BloodPressureEstimator::BloodPressureEstimator(const std::string& model_dir)
    : pImpl(new Impl(model_dir))
{
    // ファイル存在チェック等はImplのコンストラクタで行う
}

BloodPressureEstimator::~BloodPressureEstimator() = default;

std::pair<int, int> BloodPressureEstimator::estimate_bp(const std::vector<double>&, int, int, int) {
    return {0, 0};
} 

#include "bp_estimator.h"
#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <memory>
#include <fstream>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>

// 詳細タイミング計測用の構造体
struct BPTiming {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    std::string stage_name;
    
    void start(const std::string& name) {
        stage_name = name;
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    void end() {
        end_time = std::chrono::high_resolution_clock::now();
    }
    
    double get_duration_ms() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
};

struct BloodPressureEstimator::Impl {
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session sbp_session;
    Ort::Session dbp_session;
    
    // タイミング記録
    std::vector<BPTiming> timing_log;

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
    
    // タイミング計測ヘルパー関数
    void start_timing(const std::string& stage_name) {
        BPTiming timing;
        timing.start(stage_name);
        timing_log.push_back(timing);
    }
    
    void end_timing() {
        if (!timing_log.empty()) {
            timing_log.back().end();
        }
    }
    
    std::string get_timing_summary() const {
        std::stringstream ss;
        ss << "\n=== BP ESTIMATION TIMING ANALYSIS ===\n";
        
        printf("[DEBUG] BP timing_log size: %zu\n", timing_log.size());
        
        double total_time = 0.0;
        for (const auto& timing : timing_log) {
            double duration = timing.get_duration_ms();
            total_time += duration;
            ss << std::fixed << std::setprecision(2) 
               << timing.stage_name << ": " << duration << " ms\n";
            printf("[DEBUG] BP stage: %s = %.2f ms\n", timing.stage_name.c_str(), duration);
        }
        
        ss << "Total BP estimation time: " << total_time << " ms\n";
        ss << "=== BP ESTIMATION BREAKDOWN ===\n";
        
        // 各段階の割合を計算
        for (const auto& timing : timing_log) {
            double duration = timing.get_duration_ms();
            double percentage = (total_time > 0) ? (duration / total_time) * 100.0 : 0.0;
            ss << std::fixed << std::setprecision(1) 
               << timing.stage_name << ": " << percentage << "%\n";
        }
        
        std::string result = ss.str();
        printf("[DEBUG] BP timing summary length: %zu\n", result.length());
        return result;
    }

    float run(void* session_ptr, const std::vector<float>& input) {
        Ort::Session* session = session_ptr ? static_cast<Ort::Session*>(session_ptr) : &sbp_session;
        // 入力名・出力名はモデルに合わせて要調整
        const char* input_names[] = {"float_input"};
        const char* output_names[] = {"variable"};
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input.size())};
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
    : pImpl(new Impl(model_dir)), model_directory(model_dir)
{
    // ファイル存在チェック等はImplのコンストラクタで行う
}

BloodPressureEstimator::~BloodPressureEstimator() = default;

std::pair<int, int> BloodPressureEstimator::estimate_bp(const std::vector<double>& peak_times, int height, int weight, int sex) {
    pImpl->start_timing("BP Estimation Total");
    
    pImpl->start_timing("RRI Calculation and Outlier Removal");
    // rri計算・外れ値除去
    std::vector<double> rri;
    for (size_t i = 1; i < peak_times.size(); ++i) {
        double val = peak_times[i] - peak_times[i-1];
        if (val >= 0.4 && val <= 1.2) {
            rri.push_back(val);
        }
    }
    if (rri.empty()) {
        rri = {0.8, 0.8, 0.8, 0.8};
    }
    pImpl->end_timing();
    
    pImpl->start_timing("RRI Statistics Calculation");
    // rri統計量
    const size_t N = rri.size();
    double rri_mean = std::accumulate(rri.begin(), rri.end(), 0.0) / N;
    double rri_std = 0.0;
    for (double v : rri) rri_std += (v - rri_mean) * (v - rri_mean);
    rri_std = std::sqrt(rri_std / N);
    double rri_min = *std::min_element(rri.begin(), rri.end());
    double rri_max = *std::max_element(rri.begin(), rri.end());
    pImpl->end_timing();
    
    pImpl->start_timing("Feature Preparation");
    double height_m = height / 100.0;
    double bmi = weight / (height_m * height_m);
    double sex_feature = (sex == 0) ? 0.0 : 1.0;
    std::vector<float> input = {
        static_cast<float>(rri_mean),
        static_cast<float>(rri_std),
        static_cast<float>(rri_min),
        static_cast<float>(rri_max),
        static_cast<float>(bmi),
        static_cast<float>(sex_feature)
    };
    pImpl->end_timing();
    
    pImpl->start_timing("SBP ONNX Inference");
    // ONNX推論
    int sbp = static_cast<int>(std::round(pImpl->run(nullptr, input)));
    pImpl->end_timing();
    
    pImpl->start_timing("DBP ONNX Inference");
    int dbp = static_cast<int>(std::round(pImpl->run(&pImpl->dbp_session, input)));
    pImpl->end_timing();
    
    pImpl->end_timing(); // BP Estimation Total
    
    return {sbp, dbp};
}

std::string BloodPressureEstimator::get_model_dir() const {
    return model_directory;
}

std::string BloodPressureEstimator::get_timing_summary() const {
    return pImpl->get_timing_summary();
} 

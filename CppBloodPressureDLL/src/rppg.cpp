#include "rppg.h"
#include "peak_detect.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>

// MediaPipeが利用可能な場合のみ定義
#ifdef MEDIAPIPE_AVAILABLE
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <mediapipe/framework/formats/landmark.pb.h>
#include <mediapipe/framework/formats/image_frame_opencv.h>
#include <mediapipe/framework/graph_runner.h>
#endif

// 顔のROIのランドマーク番号（MediaPipeの顔メッシュ）
const std::vector<int> FACE_ROI_LANDMARKS = {
    118, 119, 120, 47, 217, 174, 196, 197, 419, 399, 437, 277,
    349, 348, 347, 280, 425, 426, 391, 393, 164, 167, 165, 206, 205, 50, 118
};

struct RPPGProcessor::Impl {
    // MediaPipe Face Mesh Graph
    std::unique_ptr<mediapipe::CalculatorGraph> graph;
    std::string input_stream_name;
    std::string output_stream_name;
    
    Impl() {
        // MediaPipe Face Mesh Graphの初期化
        mediapipe::CalculatorGraphConfig config;
        config.add_input_stream("input_video");
        config.add_output_stream("output_video");
        config.add_output_stream("face_landmarks");
        
        // Face Mesh Graphの設定
        auto* face_mesh_node = config.add_node();
        face_mesh_node->set_calculator("FaceLandmarkFrontCpu");
        face_mesh_node->add_input_stream("input_video");
        face_mesh_node->add_output_stream("output_video");
        face_mesh_node->add_output_stream("face_landmarks");
        
        // Graphの初期化
        auto status = graph->Initialize(config);
        if (!status.ok()) {
            throw std::runtime_error("Failed to initialize MediaPipe Face Mesh Graph: " + status.message());
        }
        
        input_stream_name = "input_video";
        output_stream_name = "face_landmarks";
    }
    
    std::vector<cv::Point2f> detectFaceLandmarks(const cv::Mat& frame) {
        std::vector<cv::Point2f> landmarks;
        
        // OpenCV MatをMediaPipe ImageFrameに変換
        cv::Mat rgb_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        
        auto input_frame = std::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, rgb_frame.cols, rgb_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        rgb_frame.copyTo(input_frame_mat);
        
        // Graphにフレームを送信
        size_t frame_timestamp_us = 0; // タイムスタンプ
        auto status = graph->AddPacketToInputStream(
            input_stream_name,
            mediapipe::Adopt(input_frame.release())
                .At(mediapipe::Timestamp(frame_timestamp_us)));
        
        if (!status.ok()) {
            std::cerr << "Failed to send frame to MediaPipe: " << status.message() << std::endl;
            return landmarks;
        }
        
        // 結果を取得
        mediapipe::Packet packet;
        status = graph->GetOutputStreamPoller(output_stream_name).Next(&packet);
        
        if (status.ok() && !packet.IsEmpty()) {
            const auto& face_landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
            
            // 指定されたランドマークを抽出
            for (int landmark_id : FACE_ROI_LANDMARKS) {
                if (landmark_id < face_landmarks.landmark_size()) {
                    const auto& landmark = face_landmarks.landmark(landmark_id);
                    float x = landmark.x() * frame.cols;
                    float y = landmark.y() * frame.rows;
                    landmarks.push_back(cv::Point2f(x, y));
                }
            }
        }
        
        return landmarks;
    }
};

RPPGProcessor::RPPGProcessor() : pImpl(std::make_unique<Impl>()) {
    // MediaPipe Face Mesh Graphは既にImplコンストラクタで初期化済み
}

RPPGProcessor::~RPPGProcessor() = default;

RPPGResult RPPGProcessor::processVideo(const std::string& videoPath) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        throw std::runtime_error("Cannot open video: " + videoPath);
    }
    
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;

    std::vector<std::vector<double>> skin_means; // RGB平均
    std::vector<double> timestamps;
    int frame_count = 0;
    cv::Mat frame;
    
    while (cap.read(frame)) {
        double current_time = frame_count / fps;
        frame_count++;
        
        // 顔検出とROI抽出
        std::vector<cv::Point2f> landmarks = pImpl->detectFaceLandmarks(frame);
        
        if (!landmarks.empty()) {
            // ROIマスクの作成
            cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
            std::vector<cv::Point> roi_points;
            for (const auto& pt : landmarks) {
                roi_points.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
            }
            
            if (roi_points.size() >= 3) {
                cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{roi_points}, cv::Scalar(255));
                
                // 肌色フィルタリング（YCbCr色空間）
                cv::Mat frame_ycbcr;
                cv::cvtColor(frame, frame_ycbcr, cv::COLOR_BGR2YCrCb);
                cv::Mat skin_mask;
                cv::inRange(frame_ycbcr, cv::Scalar(0, 100, 130), cv::Scalar(255, 140, 175), skin_mask);
                
                // ROIマスクと肌色マスクの組み合わせ
                cv::Mat combined_mask;
                cv::bitwise_and(mask, skin_mask, combined_mask);
                
                // ROI領域の色平均を計算
                cv::Scalar mean = cv::mean(frame, combined_mask);
                
                // 有効な平均値のみを保存
                if (mean[0] > 0 && mean[1] > 0 && mean[2] > 0) {
                    skin_means.push_back({mean[0], mean[1], mean[2]});
                    timestamps.push_back(current_time);
                }
            }
        }
    }
    
    std::vector<double> rppg_signal, time_data, peak_times;
    
    if (skin_means.size() > 30) {
        // POSアルゴリズムによるrPPG信号抽出
        int L = 30; // ウィンドウサイズ
        int frames = static_cast<int>(skin_means.size());
        
        Eigen::MatrixXd C(3, frames);
        for (int i = 0; i < frames; ++i) {
            C(0, i) = skin_means[i][0];
            C(1, i) = skin_means[i][1];
            C(2, i) = skin_means[i][2];
        }
        
        Eigen::VectorXd H = Eigen::VectorXd::Zero(frames);
        
        for (int f = 0; f < frames - L + 1; ++f) {
            Eigen::MatrixXd block = C.block(0, f, 3, L);
            Eigen::Vector3d mu_C = block.rowwise().mean();
            
            // ゼロ除算を防ぐ
            for (int i = 0; i < 3; ++i) {
                if (mu_C(i) == 0) mu_C(i) = 1e-8;
            }
            
            Eigen::MatrixXd C_normed = block.array().colwise() / mu_C.array();
            
            // POSアルゴリズムの変換行列
            Eigen::MatrixXd M(2, 3);
            M << 0, 1, -1,
                -2, 1, 1;
            
            Eigen::MatrixXd S = M * C_normed;
            
            // アルファ値の計算
            double alpha = std::sqrt(S.row(0).array().square().mean()) / 
                          (std::sqrt(S.row(1).array().square().mean()) + 1e-8);
            
            Eigen::VectorXd P = S.row(0).transpose() + alpha * S.row(1).transpose();
            
            // 正規化
            double P_mean = P.mean();
            double P_std = std::max(std::sqrt((P.array() - P_mean).square().mean()), 1e-8);
            Eigen::VectorXd P_normalized = (P.array() - P_mean) / P_std;
            
            H.segment(f, L) += P_normalized;
        }
        
        // Zスコア正規化
        double mean = H.mean();
        double stddev = std::sqrt((H.array() - mean).square().sum() / H.size());
        Eigen::VectorXd pulse_z = (H.array() - mean) / (stddev + 1e-8);
        
        // rPPG信号をstd::vectorに変換
        rppg_signal.assign(pulse_z.data(), pulse_z.data() + pulse_z.size());
        time_data = timestamps;
        
        // ピーク検出
        std::vector<size_t> peaks = find_peaks(rppg_signal, 10, 0.0);
        for (size_t idx : peaks) {
            if (idx < time_data.size()) {
                peak_times.push_back(time_data[idx]);
            }
        }
    }
    
    RPPGResult result;
    result.rppg_signal = rppg_signal;
    result.time_data = time_data;
    result.peak_times = peak_times;
    return result;
} 

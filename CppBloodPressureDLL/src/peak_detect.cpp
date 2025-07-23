#include "peak_detect.h"
#include <cmath>

std::vector<size_t> find_peaks(const std::vector<double>& signal, double min_distance, double threshold) {
    std::vector<size_t> peaks;
    for (size_t i = 1; i + 1 < signal.size(); ++i) {
        if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] > threshold) {
            // 距離条件
            if (!peaks.empty() && (i - peaks.back()) < min_distance) continue;
            peaks.push_back(i);
        }
    }
    return peaks;
} 

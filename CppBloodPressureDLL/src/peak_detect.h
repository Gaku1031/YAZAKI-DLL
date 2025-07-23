#pragma once
#include <vector>

std::vector<size_t> find_peaks(const std::vector<double>& signal, double min_distance = 10, double threshold = 0.0); 

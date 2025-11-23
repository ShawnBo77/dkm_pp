#pragma once
#ifndef DKM_THREAD_UTILS_H
#define DKM_THREAD_UTILS_H

#include <array>
#include <cstdint>
#include <thread>
#include <vector>
#include "dkm.hpp" // For clustering_parameters

namespace dkm {
namespace details {

template <typename T, size_t N>
struct ThreadArgs {
	size_t id;
	size_t begin;
	size_t end;
	const std::vector<std::array<T, N>>* data;
	const std::vector<std::array<T, N>>* means;
	std::vector<T>* distances;
	std::vector<uint32_t>* clusters;
};

template <typename T>
inline size_t determine_num_threads(const clustering_parameters<T>& parameters) {
	if (parameters.has_num_threads()) {
		return parameters.get_num_threads();
	} else {
		unsigned int n = std::thread::hardware_concurrency();
		return n == 0 ? 1 : n; // Return 1 if hardware_concurrency is not available
	}
}

} // namespace details
} // namespace dkm

#endif /* DKM_THREAD_UTILS_H */
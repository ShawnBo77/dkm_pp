#pragma once

#ifndef DKM_AVX_KMEANS_H
#define DKM_AVX_KMEANS_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

#include "dkm.hpp"

/*
DKM - A k-means implementation that is generic across variable data dimensions,
	  with a parallelized and AVX2-accelerated version.
*/
namespace dkm {
namespace details {

/*
Calculate the square of the distance between two points using AVX2 intrinsics.
*/
template <typename T, size_t N>
T distance_squared_avx(const std::array<T, N>& point_a, const std::array<T, N>& point_b) {
	// if constexpr 編譯時的條件判斷
	if constexpr (std::is_same_v<T, float> && N >= 8) {
        // For float (32-bit)
		__m256 sum_vec = _mm256_setzero_ps();
		size_t i = 0;
		// Process 8 floats at a time
		for (; i + 8 <= N; i += 8) {
			__m256 a_vec = _mm256_loadu_ps(&point_a[i]);
			__m256 b_vec = _mm256_loadu_ps(&point_b[i]);
			__m256 diff_vec = _mm256_sub_ps(a_vec, b_vec);
			sum_vec = _mm256_fmadd_ps(diff_vec, diff_vec, sum_vec); // diff_vec * diff_vec + sum_vec
		}

		// Horizontal sum
		__m128 sum_lo = _mm256_castps256_ps128(sum_vec);
		__m128 sum_hi = _mm256_extractf128_ps(sum_vec, 1);
		__m128 hsum = _mm_add_ps(sum_lo, sum_hi);
		hsum = _mm_hadd_ps(hsum, hsum);
		hsum = _mm_hadd_ps(hsum, hsum);
		T d_squared = _mm_cvtss_f32(hsum);

		// Handle remaining elements if N is not a multiple of 8
		for (; i < N; ++i) {
			auto delta = point_a[i] - point_b[i];
			d_squared += delta * delta;
		}
		return d_squared;

	} else if constexpr (std::is_same_v<T, int> && N >= 8) {
        // For int (32-bit integer)
		__m256i sum_vec = _mm256_setzero_si256();
		size_t i = 0;
		for (; i + 8 <= N; i += 8) {
			__m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&point_a[i]));
			__m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&point_b[i]));
			__m256i diff_vec = _mm256_sub_epi32(a_vec, b_vec);
			__m256i mul_vec = _mm256_mullo_epi32(diff_vec, diff_vec);
			sum_vec = _mm256_add_epi32(sum_vec, mul_vec);
		}

		// Horizontal sum
        __m128i sum_lo = _mm256_castsi256_si128(sum_vec);
        __m128i sum_hi = _mm256_extracti128_si256(sum_vec, 1);
        __m128i hsum_128 = _mm_add_epi32(sum_lo, sum_hi);
        hsum_128 = _mm_hadd_epi32(hsum_128, hsum_128);
        hsum_128 = _mm_hadd_epi32(hsum_128, hsum_128);
        T d_squared = _mm_cvtsi128_si32(hsum_128);

		for (; i < N; ++i) {
			auto delta = point_a[i] - point_b[i];
			d_squared += delta * delta;
		}

		return d_squared;

	} else {
		return distance_squared(point_a, point_b);
	}
}

/*
Calculate the smallest distance between each of the data points and any of the input means,
using AVX2 and OpenMP.
*/
template <typename T, size_t N>
std::vector<T> closest_distance_avx(
	const std::vector<std::array<T, N>>& means, const std::vector<std::array<T, N>>& data) {
	std::vector<T> distances(data.size());
    #pragma omp parallel for schedule(static)
	for (size_t i = 0; i < data.size(); ++i) {
		T closest = distance_squared_avx(data[i], means[0]);
		for (size_t j = 1; j < means.size(); ++j) {
			T distance = distance_squared_avx(data[i], means[j]);
			if (distance < closest) {
				closest = distance;
			}
		}
		distances[i] = closest;
	}
	return distances;
}

/*
Kmeans++ initialization, parallelized and using AVX2.
*/
template <typename T, size_t N>
std::vector<std::array<T, N>> random_plusplus_avx(
	const std::vector<std::array<T, N>>& data, uint32_t k, uint64_t seed) {
	assert(k > 0 && data.size() > 0);
	using input_size_t = typename std::vector<std::array<T, N>>::size_type;
	std::vector<std::array<T, N>> means;
	std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX> rand_engine(seed);

	std::uniform_int_distribution<input_size_t> uniform_generator(0, data.size() - 1);
	means.push_back(data[uniform_generator(rand_engine)]);

	for (uint32_t count = 1; count < k; ++count) {
		auto distances = details::closest_distance_avx(means, data);
#if !defined(_MSC_VER) || _MSC_VER >= 1900
		std::discrete_distribution<input_size_t> generator(distances.begin(), distances.end());
#else
		input_size_t i = 0;
		std::discrete_distribution<input_size_t> generator(
			distances.size(), 0.0, 0.0, [&distances, &i](double) { return distances[i++]; });
#endif
		means.push_back(data[generator(rand_engine)]);
	}
	return means;
}

/*
Calculate the index of the mean a particular data point is closest to, using AVX2.
*/
template <typename T, size_t N>
uint32_t closest_mean_avx(const std::array<T, N>& point, const std::vector<std::array<T, N>>& means) {
	assert(!means.empty());
	T smallest_distance = distance_squared_avx(point, means[0]);
	uint32_t index = 0;
	for (size_t i = 1; i < means.size(); ++i) {
		T distance = distance_squared_avx(point, means[i]);
		if (distance < smallest_distance) {
			smallest_distance = distance;
			index = i;
		}
	}
	return index;
}


/*
Calculate cluster assignments for each data point, using OpenMP and AVX2.
*/
template <typename T, size_t N>
std::vector<uint32_t> calculate_clusters_avx(
	const std::vector<std::array<T, N>>& data, const std::vector<std::array<T, N>>& means) {
	std::vector<uint32_t> clusters(data.size());
    #pragma omp parallel for schedule(static)
	for (size_t i = 0; i < data.size(); ++i) {
		clusters[i] = closest_mean_avx(data[i], means);
	}
	return clusters;
}

} // namespace details


/*
Parallel and SIMD-accelerated (AVX2) implementation of k-means Lloyd's algorithm.
This version is optimized for `float` data types.
*/
template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd_avx(
	const std::vector<std::array<T, N>>& data, const clustering_parameters<T>& parameters) {
	static_assert(std::is_arithmetic<T>::value && std::is_signed<T>::value,
		"kmeans_lloyd requires the template parameter T to be a signed arithmetic type (e.g. float, double, int)");
	assert(parameters.get_k() > 0);
	assert(data.size() >= parameters.get_k());

	std::random_device rand_device;
	uint64_t seed = parameters.has_random_seed() ? parameters.get_random_seed() : rand_device();
	std::vector<std::array<T, N>> means = details::random_plusplus_avx(data, parameters.get_k(), seed);

	std::vector<std::array<T, N>> old_means;
	std::vector<std::array<T, N>> old_old_means;
	std::vector<uint32_t> clusters;
	// Calculate new means until convergence is reached or we hit the maximum iteration count
	uint64_t count = 0;
	do {
		clusters = details::calculate_clusters_avx(data, means);
		old_old_means = old_means;
		old_means = means;
		means = details::calculate_means(data, clusters, old_means, parameters.get_k());
		++count;
	} while ((means != old_means && means != old_old_means)
		&& !(parameters.has_max_iteration() && count == parameters.get_max_iteration())
		&& !(parameters.has_min_delta() && details::deltas_below_limit(details::deltas(old_means, means), parameters.get_min_delta())));

	return std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>(means, clusters);

	return {means, clusters};
}

} // namespace dkm

#endif /* DKM_AVX_KMEANS_H */
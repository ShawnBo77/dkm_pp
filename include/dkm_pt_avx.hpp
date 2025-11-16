#pragma once
#ifndef DKM_PTHREAD_AVX_KMEANS_H
#define DKM_PTHREAD_AVX_KMEANS_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>
#include <pthread.h>
#include <thread>
#include <immintrin.h>
#include "dkm.hpp"   

/*
DKM - pthread 與 AVX2 整合
使用 pthread 進行平行化處理，並在計算距離時使用 AVX2 SIMD 指令集加速。
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
pthread worker：計算每個點到最近中心的歐式距離平方 (AVX2)
*/
template <typename T, size_t N>
void* worker_closest_distance_avx(void* arg) {
    auto* a = static_cast<details::ThreadArgs<T, N>*>(arg);
    const auto& data  = *a->data;
    const auto& means = *a->means;
    auto& distances   = *a->distances;

    const size_t M = means.size();
    for (size_t i = a->begin; i < a->end; ++i) {
        T best = details::distance_squared_avx<T, N>(data[i], means[0]);
        for (size_t m = 1; m < M; ++m) {
            const T d2 = details::distance_squared_avx<T, N>(data[i], means[m]);
            if (d2 < best) best = d2;
        }
        distances[i] = best;
    }
    return nullptr;
}

template <typename T, size_t N>
std::vector<T> closest_distance_pth_avx(
    const std::vector<std::array<T, N>>& means,
    const std::vector<std::array<T, N>>& data,
    const clustering_parameters<T>& parameters)
{
    const size_t n = data.size();
    std::vector<T> distances(n);
    if (n == 0) return distances;

    size_t nthr = determine_num_threads(parameters);

    const size_t chunk = (n + nthr - 1) / nthr;
    std::vector<pthread_t> threads(nthr);
    std::vector<details::ThreadArgs<T, N>> args(nthr);

    for (size_t t = 0; t < nthr; ++t) {
        auto& a = args[t];
        a.id = t;
        a.begin = t * chunk;
        a.end   = std::min(n, (t + 1) * chunk);
        a.data = &data;
        a.means = &means;
        a.distances = &distances;
        pthread_create(&threads[t], nullptr, &details::worker_closest_distance_avx<T, N>, &a);
    }
    for (size_t t = 0; t < nthr; ++t) pthread_join(threads[t], nullptr);
    return distances;
}

template <typename T, size_t N>
std::vector<std::array<T, N>> random_plusplus_pt_avx(
	const std::vector<std::array<T, N>>& data, uint32_t k, uint64_t seed, const clustering_parameters<T>& parameters) 
{
	assert(k > 0);
	assert(data.size() > 0);
	using input_size_t = typename std::array<T, N>::size_type;
	std::vector<std::array<T, N>> means;

	std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX> rand_engine(seed);

	std::uniform_int_distribution<input_size_t> uniform_generator(0, data.size() - 1);
	means.push_back(data[uniform_generator(rand_engine)]);

	for (uint32_t count = 1; count < k; ++count) {
		auto distances = details::closest_distance_pth_avx(means, data, parameters);
		
#if !defined(_MSC_VER) || _MSC_VER >= 1900
		std::discrete_distribution<input_size_t> generator(distances.begin(), distances.end());
#else
		input_size_t i = 0;
		std::discrete_distribution<input_size_t> generator(distances.size(), 0.0, 0.0, [&distances, &i](double) { return distances[i++]; });
#endif
		means.push_back(data[generator(rand_engine)]);
	}
	return means;
}

/*
pthread worker：計算每個點最近的中心索引 (AVX2 版本)
*/
template <typename T, size_t N>
void* worker_closest_mean_avx(void* arg) {
    auto* a = static_cast<details::ThreadArgs<T, N>*>(arg);
    const auto& data  = *a->data;
    const auto& means = *a->means;
    auto& clusters    = *a->clusters;

    const size_t M = means.size();
    for (size_t idx = a->begin; idx < a->end; ++idx) {
        T best = details::distance_squared_avx<T, N>(data[idx], means[0]);
        uint32_t best_id = 0;
        for (size_t m = 1; m < M; ++m) {
            const T d2 = details::distance_squared_avx<T, N>(data[idx], means[m]);
            if (d2 < best) { best = d2; best_id = static_cast<uint32_t>(m); }
        }
        clusters[idx] = best_id;
    }
    return nullptr;
}

template <typename T, size_t N>
std::vector<uint32_t> calculate_clusters_pt_avx(
    const std::vector<std::array<T, N>>& data,
    const std::vector<std::array<T, N>>& means,
    const clustering_parameters<T>& parameters)
{
    const size_t n = data.size();
    std::vector<uint32_t> clusters(n);
    if (n == 0) return clusters;

    size_t nthr = determine_num_threads(parameters);

    const size_t chunk = (n + nthr - 1) / nthr;
    std::vector<pthread_t> threads(nthr);
    std::vector<details::ThreadArgs<T, N>> args(nthr);

    for (size_t t = 0; t < nthr; ++t) {
        auto& a = args[t];
        a.id = t;
        a.begin = t * chunk;
        a.end   = std::min(n, (t + 1) * chunk);
        a.data  = &data;
        a.means = &means;
        a.clusters = &clusters;
        pthread_create(&threads[t], nullptr, &details::worker_closest_mean_avx<T, N>, &a);
    }
    for (size_t t = 0; t < nthr; ++t) pthread_join(threads[t], nullptr);
    return clusters;
}


} // namespace details

/*
主函式：kmeans_lloyd_pthread_avx (AVX2 + pthread 版本)
*/
template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>kmeans_lloyd_pt_avx(
    const std::vector<std::array<T, N>>& data,
    const clustering_parameters<T>& parameters)
{
    static_assert(std::is_arithmetic<T>::value && std::is_signed<T>::value,
        "kmeans_lloyd requires the template parameter T to be a signed arithmetic type (e.g. float, double, int)");
    assert(parameters.get_k() > 0);
    assert(data.size() >= parameters.get_k());
    std::random_device rand_device;
    uint64_t seed = parameters.has_random_seed() ? parameters.get_random_seed() : rand_device();
    
    std::vector<std::array<T, N>> means = details::random_plusplus_pt_avx<T, N>(data, parameters.get_k(), seed, parameters);

    std::vector<std::array<T, N>> old_means;
    std::vector<std::array<T, N>> old_old_means;
    std::vector<uint32_t> clusters;
    
    uint64_t count = 0;
    do {
        clusters = details::calculate_clusters_pt_avx<T, N>(data, means, parameters);
        old_old_means = old_means;
        old_means = means;
        means = details::calculate_means<T, N>(data, clusters, old_means, parameters.get_k());
        ++count;
    } while (
        (means != old_means && means != old_old_means) 
        && !(parameters.has_max_iteration() && count == parameters.get_max_iteration()) 
        && !(parameters.has_min_delta() && details::deltas_below_limit<T>(details::deltas<T, N>(old_means, means), parameters.get_min_delta()))
    );

    return std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>(means, clusters);
}

template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd_pt_avx(
    const std::vector<std::array<T, N>>& data,
    uint32_t k, 
    uint64_t max_iter = 0, 
    T min_delta = -1.0)
{
    clustering_parameters<T> parameters(k);
    if (max_iter != 0) {
        parameters.set_max_iteration(max_iter);
    }
    if (min_delta != 0) {
        parameters.set_min_delta(min_delta);   
    }
    return kmeans_lloyd_pt_avx<T, N>(data, parameters);
}

} // namespace dkm

#endif /* DKM_PTHREAD_AVX_KMEANS_H */
#pragma once
#ifndef DKM_THREAD_POOL_AVX_KMEANS_H
#define DKM_THREAD_POOL_AVX_KMEANS_H

#include "ThreadPool.hpp"
#include "dkm.hpp"
#include "dkm_avx_utils.hpp"
#include "dkm_pthread.hpp"
#include "dkm_thread_utils.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <random>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

/*
DKM - Thread Pool 與 AVX2 整合
使用 Thread Pool 進行平行化處理，並在計算距離時使用 AVX2 SIMD 指令集加速。
*/
namespace dkm {

namespace details {

template <typename T, size_t N>
void worker_closest_mean_avx_2d_float(details::ThreadArgs<T, N>* a) {
	if constexpr (N == 2 && std::is_same_v<T, float>) {
		const auto& data = *a->data;
		const auto& means = *a->means;
		auto& clusters = *a->clusters;
		const size_t num_means = means.size();
		const size_t end = a->end;
		size_t i = a->begin;

		std::vector<__m256, AlignedAllocator<__m256, 32>> pregen_means(num_means);
		for (size_t m = 0; m < num_means; ++m) {
			const float mx = means[m][0];
			const float my = means[m][1];
			pregen_means[m] = _mm256_setr_ps(mx, my, mx, my, mx, my, mx, my);
		}

		const size_t alignment_search_end = std::min(end, a->begin + 4);
		while (i < alignment_search_end && (reinterpret_cast<uintptr_t>(&data[i]) % 32 != 0)) {
			clusters[i] = closest_mean(data[i], means);
			i++;
		}

		if (i < end && (reinterpret_cast<uintptr_t>(&data[i]) % 32 == 0)) {
			for (; i + 3 < end; i += 4) {
				__m256 points_vec = _mm256_load_ps(reinterpret_cast<const float*>(&data[i]));
				__m128i final_indices_128 = find_closest_means_2d_avx_block_align(points_vec, pregen_means, num_means);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(&clusters[i]), final_indices_128);
			}
		} else {
			for (; i + 3 < end; i += 4) {
				__m128i result =
					find_closest_means_2d_avx_block(reinterpret_cast<const float*>(&data[i]), pregen_means, num_means);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(&clusters[i]), result);
			}
		}

		for (; i < end; ++i) {
			clusters[i] = closest_mean(data[i], means);
		}
	}
}

/*
Thread pool worker：計算每個點到最近中心的歐式距離平方 (AVX2)
*/
template <typename T, size_t N>
void worker_closest_distance_avx(details::ThreadArgs<T, N>* a) {
	const auto& data = *a->data;
	const auto& means = *a->means;
	auto& distances = *a->distances;

	const size_t M = means.size();
	for (size_t i = a->begin; i < a->end; ++i) {
		T best = details::distance_squared_avx<T, N>(data[i], means[0]);
		for (size_t m = 1; m < M; ++m) {
			const T d2 = details::distance_squared_avx<T, N>(data[i], means[m]);
			if (d2 < best)
				best = d2;
		}
		distances[i] = best;
	}
}

template <typename T, size_t N>
std::vector<T> closest_distance_tp_avx(const std::vector<std::array<T, N>>& means,
	const std::vector<std::array<T, N>>& data,
	const clustering_parameters<T>& parameters) {
	const size_t n = data.size();
	std::vector<T> distances(n);
	if (n == 0)
		return distances;

	size_t nthr = determine_num_threads(parameters);

	static ThreadPool pool(nthr);
	const size_t chunk = (n + nthr - 1) / nthr;
	std::vector<details::ThreadArgs<T, N>> args(nthr);

	for (size_t t = 0; t < nthr; ++t) {
		auto& a = args[t];
		a.id = t;
		a.begin = t * chunk;
		a.end = std::min(n, (t + 1) * chunk);
		a.data = &data;
		a.means = &means;
		a.distances = &distances;
		pool.enqueue([&, t]() { details::worker_closest_distance_avx<T, N>(&args[t]); });
	}

	pool.wait_all();
	return distances;
}

template <typename T, size_t N>
std::vector<std::array<T, N>> random_plusplus_tp_avx(
	const std::vector<std::array<T, N>>& data, uint32_t k, uint64_t seed, const clustering_parameters<T>& parameters) {
	assert(k > 0);
	assert(data.size() > 0);
	using input_size_t = typename std::array<T, N>::size_type;
	std::vector<std::array<T, N>> means;

	std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX> rand_engine(seed);

	std::uniform_int_distribution<input_size_t> uniform_generator(0, data.size() - 1);
	means.push_back(data[uniform_generator(rand_engine)]);

	for (uint32_t count = 1; count < k; ++count) {
		auto distances = details::closest_distance_tp_avx(means, data, parameters);

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
Thread pool worker：計算每個點最近的中心索引 (AVX2 版本)
*/
template <typename T, size_t N>
void worker_closest_mean_avx(details::ThreadArgs<T, N>* a) {
	const auto& data = *a->data;
	const auto& means = *a->means;
	auto& clusters = *a->clusters;

	const size_t M = means.size();
	for (size_t idx = a->begin; idx < a->end; ++idx) {
		T best = details::distance_squared_avx<T, N>(data[idx], means[0]);
		uint32_t best_id = 0;
		for (size_t m = 1; m < M; ++m) {
			const T d2 = details::distance_squared_avx<T, N>(data[idx], means[m]);
			if (d2 < best) {
				best = d2;
				best_id = static_cast<uint32_t>(m);
			}
		}
		clusters[idx] = best_id;
	}
}

template <typename T, size_t N>
std::vector<uint32_t> calculate_clusters_tp_avx(const std::vector<std::array<T, N>>& data,
	const std::vector<std::array<T, N>>& means,
	const clustering_parameters<T>& parameters) {
	const size_t n = data.size();
	std::vector<uint32_t> clusters(n);
	if (n == 0)
		return clusters;

	size_t nthr = determine_num_threads(parameters);

	static ThreadPool pool(nthr);
	const size_t chunk = (n + nthr - 1) / nthr;
	std::vector<details::ThreadArgs<T, N>> args(nthr);

	for (size_t t = 0; t < nthr; ++t) {
		auto& a = args[t];
		a.id = t;
		a.begin = t * chunk;
		a.end = std::min(n, (t + 1) * chunk);
		a.data = &data;
		a.means = &means;
		a.clusters = &clusters;

		if constexpr (N == 2 && std::is_same_v<T, float>) {
			pool.enqueue([&, t]() { details::worker_closest_mean_avx_2d_float<T, N>(&args[t]); });
		} else {
			pool.enqueue([&, t]() { details::worker_closest_mean_avx<T, N>(&args[t]); });
		}
	}

	pool.wait_all();
	return clusters;
}

} // namespace details

/*
主函式：kmeans_lloyd_tp_avx (AVX2 + Thread Pool 版本)
*/
template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd_tp_avx(
	const std::vector<std::array<T, N>>& data, const clustering_parameters<T>& parameters) {
	static_assert(std::is_arithmetic<T>::value && std::is_signed<T>::value,
		"kmeans_lloyd requires the template parameter T to be a signed arithmetic type (e.g. float, double, int)");
	assert(parameters.get_k() > 0);
	assert(data.size() >= parameters.get_k());
	std::random_device rand_device;
	uint64_t seed = parameters.has_random_seed() ? parameters.get_random_seed() : rand_device();

	std::vector<std::array<T, N>> means =
		details::random_plusplus_tp_avx<T, N>(data, parameters.get_k(), seed, parameters);

	std::vector<std::array<T, N>> old_means;
	std::vector<std::array<T, N>> old_old_means;
	std::vector<uint32_t> clusters;

	uint64_t count = 0;
	do {
		clusters = details::calculate_clusters_tp_avx<T, N>(data, means, parameters);
		old_old_means = old_means;
		old_means = means;
		means = details::calculate_means_avx<T, N>(data, clusters, old_means, parameters.get_k());
		++count;
	} while ((means != old_means && means != old_old_means)
		&& !(parameters.has_max_iteration() && count == parameters.get_max_iteration())
		&& !(parameters.has_min_delta()
			&& details::deltas_below_limit(details::deltas<T, N>(old_means, means), parameters.get_min_delta())));

	return std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>(means, clusters);
}

template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd_tp_avx(
	const std::vector<std::array<T, N>>& data, uint32_t k, uint64_t max_iter = 0, T min_delta = -1.0) {
	clustering_parameters<T> parameters(k);
	if (max_iter != 0) {
		parameters.set_max_iteration(max_iter);
	}
	if (min_delta != 0) {
		parameters.set_min_delta(min_delta);
	}
	return kmeans_lloyd_tp_avx<T, N>(data, parameters);
}

} // namespace dkm

#endif /* DKM_THREAD_POOL_AVX_KMEANS_H */
#pragma once
#ifndef DKM_THREAD_POOL_V2_AVX_KMEANS_H
#define DKM_THREAD_POOL_V2_AVX_KMEANS_H

#include "ThreadPool.hpp"
#include "dkm_thread_utils.hpp"
#include "dkm_tp_avx.hpp"
#include <random>
#include <tuple>
#include <vector>

/*
DKM - Thread Pool V2 與 AVX2 整合
*/
namespace dkm {
namespace details {

// 定義 block 大小（以 byte 為單位）
// constexpr size_t block_size = 16384;

/*
 * ThreadPool V2 平行版：closest_distance_tp_v2_avx
 *   – 使用較小的 block 丟進 ThreadPool 的 global queue
 *   – 重用 dkm_tp_avx.hpp 中的 worker_closest_distance_avx 函式
 */
template <typename T, size_t N>
std::vector<T> closest_distance_tp_v2_avx(const std::vector<std::array<T, N>>& means,
	const std::vector<std::array<T, N>>& data,
	const clustering_parameters<T>& parameters) {
	const size_t n = data.size();
	std::vector<T> distances(n);
	if (n == 0)
		return distances;

	size_t nthr = determine_num_threads(parameters);
	static ThreadPool pool;
	pool.reset(nthr);

	size_t points_per_block = block_size / N;
	if (points_per_block == 0)
		points_per_block = 1;
	if (points_per_block > n)
		points_per_block = n;

	const size_t num_tasks = (n + points_per_block - 1) / points_per_block;
	std::vector<ThreadArgs<T, N>> args(num_tasks);

	for (size_t t = 0; t < num_tasks; ++t) {
		auto& a = args[t];
		a.id = t;
		a.begin = t * points_per_block;
		a.end = std::min(n, (t + 1) * points_per_block);
		a.data = &data;
		a.means = &means;
		a.distances = &distances;

		pool.enqueue([&, t]() { worker_closest_distance_avx<T, N>(&args[t]); });
	}

	pool.wait_all();
	return distances;
}


template <typename T, size_t N>
std::vector<std::array<T, N>> random_plusplus_tp_v2_avx(
	const std::vector<std::array<T, N>>& data, uint32_t k, uint64_t seed, const clustering_parameters<T>& parameters) {
	assert(k > 0 && !data.empty());
	using input_size_t = typename std::array<T, N>::size_type;
	std::vector<std::array<T, N>> means;

	std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX> rand_engine(seed);

	std::uniform_int_distribution<input_size_t> uniform_generator(0, data.size() - 1);
	means.push_back(data[uniform_generator(rand_engine)]);

	for (uint32_t count = 1; count < k; ++count) {
		auto distances = details::closest_distance_tp_v2_avx<T, N>(means, data, parameters);
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
 * ThreadPool V2 平行版：calculate_clusters_tp_v2_avx
 *   – 同樣將工作切成較小 block
 */
template <typename T, size_t N>
std::vector<uint32_t> calculate_clusters_tp_v2_avx(const std::vector<std::array<T, N>>& data,
	const std::vector<std::array<T, N>>& means,
	const clustering_parameters<T>& parameters) {
	const size_t n = data.size();
	std::vector<uint32_t> clusters(n);
	if (n == 0)
		return clusters;

	size_t nthr = determine_num_threads(parameters);
	static ThreadPool pool;
	pool.reset(nthr);

	size_t points_per_block = block_size / N;
	if (points_per_block == 0)
		points_per_block = 1;
	if (points_per_block > n)
		points_per_block = n;

	const size_t num_tasks = (n + points_per_block - 1) / points_per_block;
	std::vector<ThreadArgs<T, N>> args(num_tasks);

	for (size_t t = 0; t < num_tasks; ++t) {
		auto& a = args[t];
		a.id = t;
		a.begin = t * points_per_block;
		a.end = std::min(n, (t + 1) * points_per_block);
		a.data = &data;
		a.means = &means;
		a.clusters = &clusters;

		if constexpr (N == 2 && std::is_same_v<T, float>) {
			pool.enqueue([&, t]() { worker_closest_mean_avx_2d_float<T, N>(&args[t]); });
		} else {
			pool.enqueue([&, t]() { worker_closest_mean_avx<T, N>(&args[t]); });
		}
	}
	pool.wait_all();
	return clusters;
}

} // namespace details

template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd_tp_v2_avx(
	const std::vector<std::array<T, N>>& data, const clustering_parameters<T>& parameters) {
	static_assert(
		std::is_arithmetic_v<T> && std::is_signed_v<T>, "kmeans_lloyd requires T to be a signed arithmetic type");
	assert(parameters.get_k() > 0 && data.size() >= parameters.get_k());
	std::random_device rand_device;
	uint64_t seed = parameters.has_random_seed() ? parameters.get_random_seed() : rand_device();
	std::vector<std::array<T, N>> means =
		details::random_plusplus_tp_v2_avx<T, N>(data, parameters.get_k(), seed, parameters);

	std::vector<std::array<T, N>> old_means;
	std::vector<std::array<T, N>> old_old_means;
	std::vector<uint32_t> clusters;

	uint64_t count = 0;
	do {
		clusters = details::calculate_clusters_tp_v2_avx<T, N>(data, means, parameters);
		old_old_means = old_means;
		old_means = means;
		means = details::calculate_means_avx<T, N>(data, clusters, old_means, parameters.get_k());
		++count;
	} while ((means != old_means && means != old_old_means)
		&& !(parameters.has_max_iteration() && count == parameters.get_max_iteration())
		&& !(parameters.has_min_delta()
			&& details::deltas_below_limit(details::deltas<T, N>(old_means, means), parameters.get_min_delta())));

	return {means, clusters};
}


template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd_tp_v2_avx(
	const std::vector<std::array<T, N>>& data, uint32_t k, uint64_t max_iter = 0, T min_delta = -1.0) {
	clustering_parameters<T> parameters(k);
	if (max_iter != 0)
		parameters.set_max_iteration(max_iter);
	if (min_delta >= 0)
		parameters.set_min_delta(min_delta);
	return kmeans_lloyd_tp_v2_avx<T, N>(data, parameters);
}

} // namespace dkm

#endif /* DKM_THREAD_POOL_V2_AVX_KMEANS_H */
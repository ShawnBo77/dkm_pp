#pragma once
#ifndef DKM_THREAD_POOL_KMEANS_H
#define DKM_THREAD_POOL_KMEANS_H

#include "ThreadPool.hpp"
#include "dkm.hpp"
#include "dkm_pthread.hpp"
#include "dkm_thread_utils.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <pthread.h>
#include <random>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

namespace dkm {

namespace details {

// const size_t num_threads = 8;

template <typename T, size_t N>
std::vector<T> closest_distance_tp(const std::vector<std::array<T, N>>& means,
	const std::vector<std::array<T, N>>& data,
	const clustering_parameters<T>& parameters) {
	const size_t n = data.size();
	std::vector<T> distances(n);
	if (n == 0)
		return distances;

	size_t nthr = determine_num_threads(parameters);

	// static ThreadPool：只建立於首次執行時
	static ThreadPool pool;
	pool.reset(nthr);
	const size_t chunk = (n + nthr - 1) / nthr;
	std::vector<ThreadArgs<T, N>> args(nthr);

	for (size_t t = 0; t < nthr; ++t) {
		auto& a = args[t];
		a.id = t;
		a.begin = t * chunk;
		a.end = std::min(n, (t + 1) * chunk);
		a.data = &data;
		a.means = &means;
		a.distances = &distances;

		pool.enqueue([&, t]() { worker_closest_distance<T, N>(&args[t]); });
	}

	pool.wait_all();
	return distances;
}

/*
kmaens++ 初始化找到 K 個中心作為演算法的起點。回傳 K 個中心向量組成的陣列
[kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B)
*/
template <typename T, size_t N>
std::vector<std::array<T, N>> random_plusplus_tp(
	const std::vector<std::array<T, N>>& data, uint32_t k, uint64_t seed, const clustering_parameters<T>& parameters) {
	assert(k > 0);
	assert(data.size() > 0);
	using input_size_t = typename std::array<T, N>::size_type;
	std::vector<std::array<T, N>> means;

	// Using a very simple PRBS generator, parameters selected according to
	// https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
	std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX> rand_engine(seed);

	// Select first mean at random from the set
	{
		std::uniform_int_distribution<input_size_t> uniform(0, data.size() - 1);
		means.push_back(data[uniform(rand_engine)]);
	}

	for (uint32_t count = 1; count < k; ++count) {
		// Calculate the distance to the closest mean for each data point
		auto distances = details::closest_distance_tp(means, data, parameters);

		// Pick a random point weighted by the distance from existing means
		// TODO: This might convert floating point weights to ints, distorting the distribution for small weights
#if !defined(_MSC_VER) || _MSC_VER >= 1900
		std::discrete_distribution<input_size_t> generator(distances.begin(), distances.end());
#else // MSVC++ older than 14.0
		input_size_t i = 0;
		std::discrete_distribution<input_size_t> generator(
			distances.size(), 0.0, 0.0, [&distances, &i](double) { return distances[i++]; });
#endif
		means.push_back(data[generator(rand_engine)]);
	}
	return means;
}

template <typename T, size_t N>
std::vector<uint32_t> calculate_clusters_tp(const std::vector<std::array<T, N>>& data,
	const std::vector<std::array<T, N>>& means,
	const clustering_parameters<T>& parameters) {
	const size_t n = data.size();
	std::vector<uint32_t> clusters(n);
	if (n == 0)
		return clusters;

	size_t nthr = determine_num_threads(parameters);

	// static ThreadPool：只建立於首次執行時
	static ThreadPool pool;
	pool.reset(nthr);

	const size_t chunk = (n + nthr - 1) / nthr;
	std::vector<ThreadArgs<T, N>> args(nthr);

	for (size_t t = 0; t < nthr; ++t) {
		auto& a = args[t];
		a.id = t;
		a.begin = t * chunk;
		a.end = std::min(n, (t + 1) * chunk);
		a.data = &data;
		a.means = &means;
		a.clusters = &clusters;

		pool.enqueue([&, t]() { worker_closest_mean<T, N>(&args[t]); });
	}

	pool.wait_all();
	return clusters;
}

} // namespace details

template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd_tp(
	const std::vector<std::array<T, N>>& data, const clustering_parameters<T>& parameters) {
	static_assert(std::is_arithmetic<T>::value && std::is_signed<T>::value,
		"kmeans_lloyd requires the template parameter T to be a signed arithmetic type (e.g. float, double, int)");
	assert(parameters.get_k() > 0); // k must be greater than zero
	assert(data.size() >= parameters.get_k()); // there must be at least k data points
	std::random_device rand_device;
	uint64_t seed = parameters.has_random_seed() ? parameters.get_random_seed() : rand_device();
	std::vector<std::array<T, N>> means = details::random_plusplus_tp<T, N>(data, parameters.get_k(), seed, parameters);

	std::vector<std::array<T, N>> old_means;
	std::vector<std::array<T, N>> old_old_means;
	std::vector<uint32_t> clusters;

	// Calculate new means until convergence is reached or we hit the maximum iteration count
	uint64_t count = 0;
	do {
		clusters = details::calculate_clusters_tp<T, N>(data, means, parameters);
		old_old_means = old_means;
		old_means = means;
		means = details::calculate_means<T, N>(data, clusters, old_means, parameters.get_k());
		++count;
	} while ((means != old_means && means != old_old_means)
		&& !(parameters.has_max_iteration() && count == parameters.get_max_iteration())
		&& !(parameters.has_min_delta()
			&& details::deltas_below_limit(details::deltas<T, N>(old_means, means), parameters.get_min_delta())));

	return std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>(means, clusters);
}

template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd_tp(
	const std::vector<std::array<T, N>>& data, uint32_t k, uint64_t max_iter = 0, T min_delta = -1.0) {
	clustering_parameters<T> parameters(k);

	if (max_iter != 0) {
		parameters.set_max_iteration(max_iter);
	}
	if (min_delta != 0) {
		parameters.set_min_delta(min_delta);
	}
	return kmeans_lloyd_tp<T, N>(data, parameters);
}

} // namespace dkm

#endif /* DKM_THREAD_POOL_KMEANS_H */

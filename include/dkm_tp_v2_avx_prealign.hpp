#pragma once
#ifndef DKM_TP_V2_AVX_PREALIGN_H
#define DKM_TP_V2_AVX_PREALIGN_H

#include "dkm_tp_v2_avx.hpp"
#include "dkm_avx_worker_variants.hpp"

namespace dkm {
namespace details {

template <typename T, size_t N>
std::vector<uint32_t> calculate_clusters_tp_v2_avx_prealign(const std::vector<std::array<T, N>>& data,
	const std::vector<std::array<T, N>>& means,
	const clustering_parameters<T>& parameters) {
	const size_t n = data.size();
	std::vector<uint32_t> clusters(n);
	if (n == 0) return clusters;
	size_t nthr = determine_num_threads(parameters);
	static ThreadPool pool;
	pool.reset(nthr);
	size_t points_per_block = block_size / N;
	if (points_per_block == 0) points_per_block = 1;
	if (points_per_block > n) points_per_block = n;
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
			pool.enqueue([&, t]() { worker_closest_mean_avx_2d_float_prealign<T, N>(&args[t]); });
		} else {
			pool.enqueue([&, t]() { worker_closest_mean_avx<T, N>(&args[t]); });
		}
	}
	pool.wait_all();
	return clusters;
}

} // namespace details

template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd_tp_v2_avx_prealign(
	const std::vector<std::array<T, N>>& data, const clustering_parameters<T>& parameters) {
	uint64_t seed = parameters.has_random_seed() ? parameters.get_random_seed() : std::random_device()();
	auto means = details::random_plusplus_tp_v2_avx(data, parameters.get_k(), seed, parameters);
	std::vector<uint32_t> clusters;
	std::vector<std::array<T, N>> old_means, old_old_means;
	uint64_t count = 0;
	do {
		clusters = details::calculate_clusters_tp_v2_avx_prealign<T, N>(data, means, parameters);
		old_old_means = old_means;
		old_means = means;
		means = details::calculate_means_avx<T, N>(data, clusters, old_means, parameters.get_k());
		++count;
	} while ((means != old_means && means != old_old_means)
		&& !(parameters.has_max_iteration() && count == parameters.get_max_iteration())
		&& !(parameters.has_min_delta()
			   && details::deltas_below_limit(details::deltas(old_means, means), parameters.get_min_delta())));
	return {means, clusters};
}
} // namespace dkm
#endif // DKM_TP_V2_AVX_PREALIGN_H
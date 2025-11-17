#pragma once
#ifndef DKM_PTHREAD_AVX_KMEANS_H
#define DKM_PTHREAD_AVX_KMEANS_H

#include "dkm.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
// #include <iostream>
#include <pthread.h>
#include <random>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

// ==========================================================
// Aligned Allocator (with rebind for STL compatibility)
// ==========================================================
template <typename T, size_t Alignment>
class AlignedAllocator {
	public:
	using value_type = T;
	static constexpr size_t alignment = Alignment;

	template <class U>
	struct rebind {
		using other = AlignedAllocator<U, Alignment>;
	};

	AlignedAllocator() noexcept = default;

	template <class U>
	constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

	T* allocate(size_t n) {
		if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
			throw std::bad_alloc();
		}
		size_t bytes = n * sizeof(T);
		void* p = nullptr;
#if defined(_MSC_VER)
		p = _aligned_malloc(bytes, Alignment);
#else
		p = aligned_alloc(Alignment, bytes);
#endif
		if (!p) {
			throw std::bad_alloc();
		}
		return static_cast<T*>(p);
	}

	void deallocate(T* p, size_t /* n */) noexcept {
#if defined(_MSC_VER)
		_aligned_free(p);
#else
		free(p);
#endif
	}
};

template <class T, size_t N1, class U, size_t N2>
constexpr bool operator==(const AlignedAllocator<T, N1>&, const AlignedAllocator<U, N2>&) noexcept {
	return N1 == N2;
}

template <class T, size_t N1, class U, size_t N2>
constexpr bool operator!=(const AlignedAllocator<T, N1>&, const AlignedAllocator<U, N2>&) noexcept {
	return N1 != N2;
}

/*
DKM - pthread 與 AVX2 整合
使用 pthread 進行平行化處理，並在計算距離時使用 AVX2 SIMD 指令集加速。
*/
namespace dkm {

namespace details {

// SIMD 工具函式
namespace simd_utils {
// 水平加總 __m256 (8 個 float)
inline float horizontal_add_ps(__m256 v) {
	__m128 vlow = _mm256_castps256_ps128(v);
	__m128 vhigh = _mm256_extractf128_ps(v, 1);
	vlow = _mm_add_ps(vlow, vhigh);
	__m128 shuf = _mm_movehdup_ps(vlow);
	__m128 sums = _mm_add_ps(vlow, shuf);
	shuf = _mm_movehl_ps(shuf, sums);
	sums = _mm_add_ss(sums, shuf);
	return _mm_cvtss_f32(sums);
}
} // namespace simd_utils

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
		// 每次處理 8 個 float
		for (; i + 7 < N; i += 8) {
			__m256 a_vec = _mm256_loadu_ps(&point_a[i]);
			__m256 b_vec = _mm256_loadu_ps(&point_b[i]);
			__m256 diff_vec = _mm256_sub_ps(a_vec, b_vec);
			sum_vec = _mm256_fmadd_ps(diff_vec, diff_vec, sum_vec);
		}
		T d_squared = simd_utils::horizontal_add_ps(sum_vec);

		// 處理最後剩下的不足 8 個的元素
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
pthread worker: 為 N=2 的 float 資料特化的 AVX 版本。
一次處理 4 個 2D 點。
*/

inline __m128i find_closest_means_2d_avx_block_align(
	__m256 points_vec, const std::vector<__m256, AlignedAllocator<__m256, 32>>& pregen_means, size_t num_means) {
	__m256 min_dist_sq = _mm256_set1_ps(std::numeric_limits<float>::max());
	__m256i best_indices = _mm256_setzero_si256();

	for (size_t m = 0; m < num_means; ++m) {
		__m256 mean_vec = _mm256_load_ps(reinterpret_cast<const float*>(&pregen_means[m]));

		__m256 diff = _mm256_sub_ps(points_vec, mean_vec);
		__m256 diff_sq = _mm256_mul_ps(diff, diff);

		__m256 shuf_sq = _mm256_permute_ps(diff_sq, 0b10110001);
		__m256 dist_sq = _mm256_add_ps(diff_sq, shuf_sq);

		__m256 mask = _mm256_cmp_ps(dist_sq, min_dist_sq, _CMP_LT_OQ);
		min_dist_sq = _mm256_min_ps(min_dist_sq, dist_sq);

		__m256i current_indices = _mm256_set1_epi32(static_cast<int>(m));
		best_indices = _mm256_blendv_epi8(best_indices, current_indices, _mm256_castps_si256(mask));
	}

	// 打包索引: 從 [id0, id0, id1, id1, id2, id2, id3, id3] 提取 [id0, id1, id2, id3]
	const __m256i pack_mask = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
	__m256i packed_indices = _mm256_permutevar8x32_epi32(best_indices, pack_mask);

	// 返回 4 個索引的 128-bit 結果
	return _mm256_castsi256_si128(packed_indices);
}

inline __m128i find_closest_means_2d_avx_block(
	const float* data_ptr, const std::vector<__m256, AlignedAllocator<__m256, 32>>& pregen_means, size_t num_means) {
	__m256 points_vec = _mm256_loadu_ps(data_ptr);
	__m256 min_dist_sq = _mm256_set1_ps(std::numeric_limits<float>::max());
	__m256i best_indices = _mm256_setzero_si256();

	for (size_t m = 0; m < num_means; ++m) {
		__m256 mean_vec = _mm256_load_ps(reinterpret_cast<const float*>(&pregen_means[m]));

		__m256 diff = _mm256_sub_ps(points_vec, mean_vec);
		__m256 diff_sq = _mm256_mul_ps(diff, diff);

		__m256 shuf_sq = _mm256_permute_ps(diff_sq, 0b10110001);
		__m256 dist_sq = _mm256_add_ps(diff_sq, shuf_sq);

		__m256 mask = _mm256_cmp_ps(dist_sq, min_dist_sq, _CMP_LT_OQ);
		min_dist_sq = _mm256_min_ps(min_dist_sq, dist_sq);

		__m256i current_indices = _mm256_set1_epi32(static_cast<int>(m));
		best_indices = _mm256_blendv_epi8(best_indices, current_indices, _mm256_castps_si256(mask));
	}

	const __m256i pack_mask = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
	__m256i packed_indices = _mm256_permutevar8x32_epi32(best_indices, pack_mask);

	return _mm256_castsi256_si128(packed_indices);
}

template <typename T, size_t N>
void* worker_closest_mean_avx_2d_float(void* arg) {
	if constexpr (N == 2 && std::is_same_v<T, float>) {
		auto* a = static_cast<details::ThreadArgs<T, N>*>(arg);
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

        // 開頭沒對齊的部分
		while (i < end && (reinterpret_cast<uintptr_t>(&data[i]) % 32 != 0)) {
			clusters[i] = closest_mean(data[i], means);
			i++;
		}

		// 對齊的 AVX 計算
		for (; i + 3 < end; i += 4) {
			// 對齊載入 4 個點的資料
			__m256 points_vec = _mm256_load_ps(reinterpret_cast<const float*>(&data[i]));
			// 計算 4 個點最近的 mean
			__m128i final_indices_128 = find_closest_means_2d_avx_block_align(points_vec, pregen_means, num_means);
			// 將結果儲存回記憶體
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&clusters[i]), final_indices_128);
		}

		// 全部沒對齊
		// for (; i + 3 < end; i += 4) {
		//     __m128i result = find_closest_means_2d_avx_block(
		//         reinterpret_cast<const float*>(&data[i]), pregen_means, num_means);
		//     _mm_storeu_si128(reinterpret_cast<__m128i*>(&clusters[i]), result);
		// }

		// 剩餘部分
		for (; i < end; ++i) {
			clusters[i] = closest_mean(data[i], means);
		}
		return nullptr;
	}
	return nullptr;
}

// 預先對齊資料
template <typename T, size_t N>
void* worker_closest_mean_avx_2d_float_align(void* arg) {
	if constexpr (N == 2 && std::is_same_v<T, float>) {
		auto* a = static_cast<details::ThreadArgs<T, N>*>(arg);
		const size_t start_idx = a->begin;
		const size_t end_idx = a->end;

		// 如果沒有被分配到工作，直接返回
		if (start_idx >= end_idx) {
			return nullptr;
		}

		const auto& original_data = *a->data;
		const auto& means = *a->means;
		auto& clusters = *a->clusters;
		const size_t num_means = means.size();
		const size_t task_size = end_idx - start_idx;

		std::vector<std::array<T, N>, AlignedAllocator<std::array<T, N>, 32>> aligned_task_data(task_size);

		std::copy(original_data.begin() + start_idx, original_data.begin() + end_idx, aligned_task_data.begin());

		const auto& data = aligned_task_data;

		std::vector<__m256, AlignedAllocator<__m256, 32>> pregen_means(num_means);
		for (size_t m = 0; m < num_means; ++m) {
			const float mx = means[m][0];
			const float my = means[m][1];
			pregen_means[m] = _mm256_setr_ps(mx, my, mx, my, mx, my, mx, my);
		}

		const __m256i pack_mask = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);

		for (size_t i = 0; i + 3 < task_size; i += 4) {
			__m256 points_vec = _mm256_load_ps(reinterpret_cast<const float*>(&data[i]));
			__m128i final_indices_128 = find_closest_means_2d_avx_block_align(points_vec, pregen_means, num_means);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&clusters[start_idx + i]), final_indices_128);
		}

		size_t last_avx_idx = (task_size / 4) * 4;
		for (size_t i = last_avx_idx; i < task_size; ++i) {
			clusters[start_idx + i] = closest_mean(data[i], means);
		}
		return nullptr;
	}
	return nullptr;
}

/*
pthread worker：計算每個點到最近中心的歐式距離平方 (AVX2)
*/
template <typename T, size_t N>
void* worker_closest_distance_avx(void* arg) {
	auto* a = static_cast<details::ThreadArgs<T, N>*>(arg);
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
	return nullptr;
}

template <typename T, size_t N>
std::vector<T> closest_distance_pth_avx(const std::vector<std::array<T, N>>& means,
	const std::vector<std::array<T, N>>& data,
	const clustering_parameters<T>& parameters) {
	const size_t n = data.size();
	std::vector<T> distances(n);
	if (n == 0)
		return distances;

	size_t nthr = determine_num_threads(parameters);

	const size_t chunk = (n + nthr - 1) / nthr;
	std::vector<pthread_t> threads(nthr);
	std::vector<details::ThreadArgs<T, N>> args(nthr);

	for (size_t t = 0; t < nthr; ++t) {
		auto& a = args[t];
		a.id = t;
		a.begin = t * chunk;
		a.end = std::min(n, (t + 1) * chunk);
		a.data = &data;
		a.means = &means;
		a.distances = &distances;
		pthread_create(&threads[t], nullptr, &details::worker_closest_distance_avx<T, N>, &a);
	}
	for (size_t t = 0; t < nthr; ++t)
		pthread_join(threads[t], nullptr);
	return distances;
}

template <typename T, size_t N>
std::vector<std::array<T, N>> random_plusplus_pt_avx(
	const std::vector<std::array<T, N>>& data, uint32_t k, uint64_t seed, const clustering_parameters<T>& parameters) {
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
		std::discrete_distribution<input_size_t> generator(
			distances.size(), 0.0, 0.0, [&distances, &i](double) { return distances[i++]; });
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
	return nullptr;
}

template <typename T, size_t N>
std::vector<uint32_t> calculate_clusters_pt_avx(const std::vector<std::array<T, N>>& data,
	const std::vector<std::array<T, N>>& means,
	const clustering_parameters<T>& parameters) {
	const size_t n = data.size();
	std::vector<uint32_t> clusters(n);
	if (n == 0)
		return clusters;

	size_t nthr = determine_num_threads(parameters);

	const size_t chunk = (n + nthr - 1) / nthr;
	std::vector<pthread_t> threads(nthr);
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
			pthread_create(&threads[t], nullptr, &details::worker_closest_mean_avx_2d_float<T, N>, &a);
		} else {
			pthread_create(&threads[t], nullptr, &details::worker_closest_mean_avx<T, N>, &a);
		}
	}
	for (size_t t = 0; t < nthr; ++t)
		pthread_join(threads[t], nullptr);
	return clusters;
}


} // namespace details

/*
主函式：kmeans_lloyd_pthread_avx (AVX2 + pthread 版本)
*/
template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd_pt_avx(
	const std::vector<std::array<T, N>>& data, const clustering_parameters<T>& parameters) {
	static_assert(std::is_arithmetic<T>::value && std::is_signed<T>::value,
		"kmeans_lloyd requires the template parameter T to be a signed arithmetic type (e.g. float, double, int)");
	assert(parameters.get_k() > 0);
	assert(data.size() >= parameters.get_k());
	std::random_device rand_device;
	uint64_t seed = parameters.has_random_seed() ? parameters.get_random_seed() : rand_device();

	std::vector<std::array<T, N>> means =
		details::random_plusplus_pt_avx<T, N>(data, parameters.get_k(), seed, parameters);

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
	} while ((means != old_means && means != old_old_means)
		&& !(parameters.has_max_iteration() && count == parameters.get_max_iteration())
		&& !(parameters.has_min_delta()
			&& details::deltas_below_limit<T>(details::deltas<T, N>(old_means, means), parameters.get_min_delta())));

	return std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>(means, clusters);
}

template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd_pt_avx(
	const std::vector<std::array<T, N>>& data, uint32_t k, uint64_t max_iter = 0, T min_delta = -1.0) {
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
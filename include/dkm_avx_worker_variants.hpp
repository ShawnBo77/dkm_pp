#pragma once
#ifndef DKM_AVX_WORKER_VARIANTS_H
#define DKM_AVX_WORKER_VARIANTS_H

#include "dkm_thread_utils.hpp"
#include "dkm_avx_utils.hpp"
#include <vector>
#include <algorithm>

namespace dkm {
namespace details {

// ===================================================================
// 使用 Unaligned Load 的 Worker
// ===================================================================
template <typename T, size_t N>
void worker_closest_mean_avx_2d_float_unalign(details::ThreadArgs<T, N>* a) {
	if constexpr (N == 2 && std::is_same_v<T, float>) {
		const auto& data = *a->data;
		const auto& means = *a->means;
		auto& clusters = *a->clusters;
		const size_t num_means = means.size();
		const size_t end = a->end;
		size_t i = a->begin;

		std::vector<__m256, AlignedAllocator<__m256, 32>> pregen_means(num_means);
		for (size_t m = 0; m < num_means; ++m) {
			pregen_means[m] = _mm256_setr_ps(means[m][0], means[m][1], means[m][0], means[m][1], means[m][0], means[m][1], means[m][0], means[m][1]);
		}

		for (; i + 3 < end; i += 4) {
			__m128i result = find_closest_means_2d_avx_block(
				reinterpret_cast<const float*>(&data[i]), pregen_means, num_means);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&clusters[i]), result);
		}

		// 處理剩餘部分
		for (; i < end; ++i) {
			clusters[i] = closest_mean(data[i], means);
		}
	}
}

// ===================================================================
// 預先對齊資料的 Worker
// ===================================================================
template <typename T, size_t N>
void worker_closest_mean_avx_2d_float_prealign(details::ThreadArgs<T, N>* a) {
	if constexpr (N == 2 && std::is_same_v<T, float>) {
		const size_t start_idx = a->begin;
		const size_t end_idx = a->end;

		if (start_idx >= end_idx) {
			return;
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
			pregen_means[m] = _mm256_setr_ps(means[m][0], means[m][1], means[m][0], means[m][1], means[m][0], means[m][1], means[m][0], means[m][1]);
		}

		// aligned load
		for (size_t i = 0; i + 3 < task_size; i += 4) {
			__m256 points_vec = _mm256_load_ps(reinterpret_cast<const float*>(&data[i]));
			__m128i final_indices_128 = find_closest_means_2d_avx_block_align(points_vec, pregen_means, num_means);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&clusters[start_idx + i]), final_indices_128);
		}

		// 處理剩餘部分
		size_t last_avx_idx = (task_size / 4) * 4;
		for (size_t i = last_avx_idx; i < task_size; ++i) {
			clusters[start_idx + i] = closest_mean(data[i], means);
		}
	}
}

} // namespace details
} // namespace dkm

#endif // DKM_AVX_WORKER_VARIANTS_H
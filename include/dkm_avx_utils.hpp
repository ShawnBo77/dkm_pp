#pragma once
#ifndef DKM_AVX_UTILS_H
#define DKM_AVX_UTILS_H

#include "dkm.hpp"
#include <array>
#include <cstdint>
#include <immintrin.h>
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
		if (bytes == 0)
			return nullptr;
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
std::vector<std::array<T, N>> calculate_means_avx(const std::vector<std::array<T, N>>& data,
	const std::vector<uint32_t>& clusters,
	const std::vector<std::array<T, N>>& old_means,
	uint32_t k) {
	if constexpr (std::is_same_v<T, float> && N >= 8) {
		// Accumulation
		std::vector<std::array<T, N>> means(k);
		std::vector<T> counts(k, T(0));
		for (size_t i = 0; i < data.size(); ++i) {
			const uint32_t cluster_idx = clusters[i];
			counts[cluster_idx] += 1;
			auto& mean = means[cluster_idx];
			const auto& point = data[i];

			size_t j = 0;
			for (; j + 7 < N; j += 8) {
				__m256 mean_vec = _mm256_loadu_ps(&mean[j]);
				__m256 point_vec = _mm256_loadu_ps(&point[j]);
				__m256 sum_vec = _mm256_add_ps(mean_vec, point_vec);
				_mm256_storeu_ps(&mean[j], sum_vec);
			}
			for (; j < N; ++j) {
				mean[j] += point[j];
			}
		}

		// Averaging
		for (uint32_t i = 0; i < k; ++i) {
			if (counts[i] == 0) {
				means[i] = old_means[i];
			} else {
				__m256 count_vec = _mm256_set1_ps(static_cast<float>(counts[i]));
				size_t j = 0;
				for (; j + 7 < N; j += 8) {
					__m256 mean_vec = _mm256_loadu_ps(&means[i][j]);
					__m256 avg_vec = _mm256_div_ps(mean_vec, count_vec);
					_mm256_storeu_ps(&means[i][j], avg_vec);
				}
				for (; j < N; ++j) {
					means[i][j] /= counts[i];
				}
			}
		}
		return means;

	} else {
		return dkm::details::calculate_means(data, clusters, old_means, k);
	}
}

} // namespace details
} // namespace dkm

#endif // DKM_AVX_UTILS_H
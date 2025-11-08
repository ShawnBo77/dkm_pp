// clang-format disabled because clang-format doesn't format lest's macros correctly
// clang-format off
/*
Test cases for dkm.hpp

This is just simple test harness without any external dependencies.
*/

#include "../../include/dkm.hpp"
#include "../../include/dkm_parallel.hpp"
// #include "../../include/dkm_thread.hpp"
#include "../../include/dkm_avx.hpp"
#include "../../include/dkm_utils.hpp"
#include "lest.hpp"

#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>
#include <tuple>
#include <iostream>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wmissing-braces"
#endif

constexpr uint64_t random_seed_value = 7;

template <typename T, size_t N>
void verify_clustering_results(
	lest::env& $, // 傳入 lest 環境
	const std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>& standard,
	const std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>& to_verify) {
	
	auto& serial_means = std::get<0>(standard);
	auto& serial_clusters = std::get<1>(standard);
	auto& verified_means = std::get<0>(to_verify);
	auto& verified_clusters = std::get<1>(to_verify);

	EXPECT(serial_means.size() == verified_means.size());

	// 解決標籤排列問題
	std::vector<int> serial_to_verified_map(serial_means.size(), -1);
	std::vector<bool> verified_mean_is_matched(verified_means.size(), false);

	for (size_t i = 0; i < serial_means.size(); ++i) {
		T min_dist_sq = -1;
		int best_match_idx = -1;

		for (size_t j = 0; j < verified_means.size(); ++j) {
			if (verified_mean_is_matched[j]) continue;

			T dist_sq = dkm::details::distance_squared(serial_means[i], verified_means[j]);
			if (best_match_idx == -1 || dist_sq < min_dist_sq) {
				min_dist_sq = dist_sq;
				best_match_idx = j;
			}
		}
		
		serial_to_verified_map[i] = best_match_idx;
		verified_mean_is_matched[best_match_idx] = true;

		// 驗證中心點座標
		for (size_t dim = 0; dim < N; ++dim) {
			EXPECT(serial_means[i][dim] == lest::approx(verified_means[best_match_idx][dim]));
		}
	}

	// 驗證群集分配
	for (size_t i = 0; i < serial_clusters.size(); ++i) {
		uint32_t serial_label = serial_clusters[i];
		uint32_t verified_label = verified_clusters[i];
		EXPECT(serial_to_verified_map[serial_label] == (int)verified_label);
	}
}

const lest::test specification[] = {
	CASE("Small 2D dataset is successfully segmented into 3 clusters",) {
		SETUP("Small 2D dataset") {
			std::vector<std::array<float, 2>> data{
				{18.789, 19.684 },
				{-41.478, -19.799},
				{-22.410, -6.794},
				{-29.411  , -8.416},
				{194.874, 6.187},
				{86.881, 34.023},
				{125.640, 24.364},
				{14.900, 29.114 },
				{15.082, 23.051},
				{-24.638, -7.013},
				{-26.608, -23.007},
				{-31.118, -11.876},
				{-24.734, -3.788 },
				{133.423, 23.644},
				{14.346, 21.789},
				{16.875, 23.290},
				{132.308, -0.032}
			};

			// means: [17,27], [-27, -12], [128, 10]
			dkm::clustering_parameters<float> parameters(3);
			parameters.set_random_seed(random_seed_value);
			
			SECTION("Distance squared calculated correctly") {
				EXPECT(dkm::details::distance_squared(data[0], data[1]) == lest::approx(5191.02f));
				EXPECT(dkm::details::distance_squared(data[1], data[2]) == lest::approx(532.719f));
			}
			
			SECTION("Initial means picked correctly") {
				auto means = dkm::details::random_plusplus(data, parameters.get_k(), parameters.get_random_seed());
				std::vector<std::array<float, 2>> expected_means{{15.082f, 23.051f}, {133.423f, 23.644f}, {-24.734f, -3.788f}};
				EXPECT(means.size() == 3u);
				EXPECT(means == expected_means);
			}
			
			SECTION("K-means calculated correctly via Lloyds method") {
				auto means_clusters = dkm::kmeans_lloyd(data, parameters);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);
				// verify results
				std::vector<std::array<float, 2>> expected_means{{15.9984f, 23.3856f}, {134.625f, 17.6372f}, {-28.6281f, -11.5276f}};
				EXPECT(means.size() == 3u);
				for (size_t i = 0; i < means.size(); ++i) {
					for (size_t j = 0; j < means[i].size(); ++j) {
						EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
					}
				}
				std::vector<uint32_t> expected_clusters = { 0, 2, 2, 2, 1, 1, 1, 0, 0, 2, 2, 2, 2, 1, 0, 0, 1};
				EXPECT(clusters.size() == data.size());
				EXPECT(clusters == expected_clusters);
			}

			SECTION("K-means calculated correctly via parallel Lloyds method") {
				auto means_clusters = dkm::kmeans_lloyd_parallel(data, parameters);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);
				// verify results
				EXPECT(means.size() == 3u);
				EXPECT(clusters.size() == data.size());
				std::vector<std::array<float, 2>> expected_means{{15.9984f, 23.3856f}, {134.625f, 17.6372f}, {-28.6281f, -11.5276f}};
				EXPECT(means.size() == 3u);
				for (size_t i = 0; i < means.size(); ++i) {
					for (size_t j = 0; j < means[i].size(); ++j) {
						EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
					}
				}
				std::vector<uint32_t> expected_clusters = { 0, 2, 2, 2, 1, 1, 1, 0, 0, 2, 2, 2, 2, 1, 0, 0, 1};
				EXPECT(clusters.size() == data.size());
				EXPECT(clusters == expected_clusters);
			}

			// SECTION("K-means calculated correctly via parallel Lloyds method (thread)") {
			// 	auto means_clusters = dkm::kmeans_lloyd_thread(data, parameters); // 呼叫 Pthread 版本
			// 	auto means = std::get<0>(means_clusters);
			// 	auto clusters = std::get<1>(means_clusters);

			// 	EXPECT(means.size() == 3u);
			// 	EXPECT(clusters.size() == data.size());
			// 	std::vector<std::array<float, 2>> expected_means{{15.9984f, 23.3856f}, {134.625f, 17.6372f}, {-28.6281f, -11.5276f}};
			// 	EXPECT(means.size() == 3u);
			// 	for (size_t i = 0; i < means.size(); ++i) {
			// 		for (size_t j = 0; j < means[i].size(); ++j) {
			// 			EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
			// 		}
			// 	}
			// 	std::vector<uint32_t> expected_clusters = { 0, 2, 2, 2, 1, 1, 1, 0, 0, 2, 2, 2, 2, 1, 0, 0, 1};
			// 	EXPECT(clusters.size() == data.size());
			// 	EXPECT(clusters == expected_clusters);
			// }

			SECTION("K-means calculated correctly via parallel Lloyds method (AVX2)") {
				auto means_clusters = dkm::kmeans_lloyd_avx(data, parameters);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);
				// verify results (will use scalar fallback for N=2, so should be identical)
				EXPECT(means.size() == 3u);
				EXPECT(clusters.size() == data.size());
				std::vector<std::array<float, 2>> expected_means{{15.9984f, 23.3856f}, {134.625f, 17.6372f}, {-28.6281f, -11.5276f}};
				EXPECT(means.size() == 3u);
				for (size_t i = 0; i < means.size(); ++i) {
					for (size_t j = 0; j < means[i].size(); ++j) {
						EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
					}
				}
				std::vector<uint32_t> expected_clusters = { 0, 2, 2, 2, 1, 1, 1, 0, 0, 2, 2, 2, 2, 1, 0, 0, 1};
				EXPECT(clusters.size() == data.size());
				EXPECT(clusters == expected_clusters);
			}
		}
	},

	CASE("Test AVX implementation with varied N=8 float data",) {
        SETUP("8D float dataset with varied values") {
            std::vector<std::array<float, 8>> data{
                {1.1f, -2.2f, 3.3f, -4.4f, 5.5f, -6.6f, 7.7f, -8.8f},
                {1.2f, -2.3f, 3.4f, -4.5f, 5.6f, -6.7f, 7.8f, -8.9f},
                {-10.1f, 11.2f, -12.3f, 13.4f, -14.5f, 15.6f, -16.7f, 17.8f},
                {-10.2f, 11.3f, -12.4f, 13.5f, -14.6f, 15.7f, -16.8f, 17.9f},
                {100.5f, 101.5f, 102.5f, 103.5f, 104.5f, 105.5f, 106.5f, 107.5f},
                {100.6f, 101.6f, 102.6f, 103.6f, 104.6f, 105.6f, 106.6f, 107.6f}
            };
            dkm::clustering_parameters<float> parameters(3);
			parameters.set_random_seed(random_seed_value);

            SECTION("AVX version produces same result as scalar version") {
                auto serial_res = dkm::kmeans_lloyd(data, parameters);
                auto avx_res = dkm::kmeans_lloyd_avx(data, parameters);

                auto serial_means = std::get<0>(serial_res);
                auto avx_means = std::get<0>(avx_res);
                auto serial_clusters = std::get<1>(serial_res);
                auto avx_clusters = std::get<1>(avx_res);

                EXPECT(avx_means.size() == serial_means.size());
                EXPECT(avx_clusters == serial_clusters);
                for(size_t i = 0; i < serial_means.size(); ++i) {
                    for(size_t j = 0; j < 8; ++j) {
                        EXPECT(serial_means[i][j] == lest::approx(avx_means[i][j]));
                    }
                }
            }
        }
    },

	CASE("Test AVX implementation with dimension not a multiple of 8 (N=10)",) {
		SETUP("10D float dataset to test remainder handling") {
			std::vector<std::array<float, 10>> data{
                {1.1f, -2.2f, 3.3f, -4.4f, 5.5f, -6.6f, 7.7f, -8.8f, 9.9f, -10.1f},
                {1.2f, -2.3f, 3.4f, -4.5f, 5.6f, -6.7f, 7.8f, -8.9f, 9.8f, -10.2f},
                {-10.1f, 11.2f, -12.3f, 13.4f, -14.5f, 15.6f, -16.7f, 17.8f, -18.9f, 19.0f},
                {-10.2f, 11.3f, -12.4f, 13.5f, -14.6f, 15.7f, -16.8f, 17.9f, -18.8f, 19.1f}
            };
			dkm::clustering_parameters<float> parameters(2);
			parameters.set_random_seed(random_seed_value);

			SECTION("AVX version (N=10) produces same result as scalar version") {
                auto serial_res = dkm::kmeans_lloyd(data, parameters);
                auto avx_res = dkm::kmeans_lloyd_avx(data, parameters);

                auto serial_means = std::get<0>(serial_res);
                auto avx_means = std::get<0>(avx_res);
                auto serial_clusters = std::get<1>(serial_res);
                auto avx_clusters = std::get<1>(avx_res);

                EXPECT(avx_means.size() == serial_means.size());
                EXPECT(avx_clusters == serial_clusters);
                for(size_t i = 0; i < serial_means.size(); ++i) {
                    for(size_t j = 0; j < 10; ++j) { // 檢查所有10個維度
                        EXPECT(serial_means[i][j] == lest::approx(avx_means[i][j]));
                    }
                }
			}
		}
	},

	CASE("Test AVX implementation for integer types (N=8)",) {
		SETUP("8D integer dataset") {
			std::vector<std::array<int, 8>> data{
				{1, 2, 3, 4, 5, 6, 7, 8},
				{-1, -2, -3, -4, -5, -6, -7, -8},
				{10, 20, 30, 40, 50, 60, 70, 80},
				{-10, -20, -30, -40, -50, -60, -70, -80},
				{101, 102, 103, 104, 105, 106, 107, 108},
				{-101, -102, -103, -104, -105, -106, -107, -108}
			};
			dkm::clustering_parameters<int> parameters(3);
			parameters.set_random_seed(random_seed_value);

			SECTION("AVX integer version produces same result as scalar version") {
				auto serial_res = dkm::kmeans_lloyd(data, parameters);
				auto avx_res = dkm::kmeans_lloyd_avx(data, parameters);
				
				auto serial_means = std::get<0>(serial_res);
                auto avx_means = std::get<0>(avx_res);
                auto serial_clusters = std::get<1>(serial_res);
                auto avx_clusters = std::get<1>(avx_res);

                EXPECT(avx_means.size() == serial_means.size());
                EXPECT(avx_clusters == serial_clusters);
                for(size_t i = 0; i < serial_means.size(); ++i) {
                    for(size_t j = 0; j < 8; ++j) {
                        EXPECT(serial_means[i][j] == avx_means[i][j]);
                    }
                }
			}
		}
	},

	CASE("Verify correctness with benchmark dataset: iris.data.csv",) {
		// lest::env& $ 參數是 lest 框架自動傳入的
		SETUP(lest::env& $) {
			auto data = dkm::load_csv<float, 2>("../bench/iris.data.csv");
			dkm::clustering_parameters<float> parameters(3);
			parameters.set_random_seed(42);

			auto serial_res = dkm::kmeans_lloyd(data, parameters);
			
			SECTION("Parallel version matches serial version") {
				auto parallel_res = dkm::kmeans_lloyd_parallel(data, parameters);
				verify_clustering_results($, serial_res, parallel_res);
			}

			SECTION("AVX version matches serial version") {
				auto avx_res = dkm::kmeans_lloyd_avx(data, parameters);
				verify_clustering_results($, serial_res, avx_res);
			}
		}
	},

	CASE("Verify correctness with benchmark dataset: s1.data.csv",) {
		SETUP(lest::env& $) {
			auto data = dkm::load_csv<float, 2>("../bench/s1.data.csv");
			dkm::clustering_parameters<float> parameters(15);
			parameters.set_random_seed(42);

			auto serial_res = dkm::kmeans_lloyd(data, parameters);
			
			SECTION("Parallel version matches serial version") {
				auto parallel_res = dkm::kmeans_lloyd_parallel(data, parameters);
				verify_clustering_results($, serial_res, parallel_res);
			}

			SECTION("AVX version matches serial version") {
				auto avx_res = dkm::kmeans_lloyd_avx(data, parameters);
				verify_clustering_results($, serial_res, avx_res);
			}
		}
	},

	CASE("Verify correctness with benchmark dataset: birch3.data.csv",) {
		SETUP(lest::env& $) {
			auto data = dkm::load_csv<float, 2>("../bench/birch3.data.csv");
			dkm::clustering_parameters<float> parameters(100);
			parameters.set_random_seed(42);

			auto serial_res = dkm::kmeans_lloyd(data, parameters);
			
			SECTION("Parallel version matches serial version") {
				auto parallel_res = dkm::kmeans_lloyd_parallel(data, parameters);
				verify_clustering_results($, serial_res, parallel_res);
			}

			SECTION("AVX version matches serial version") {
				auto avx_res = dkm::kmeans_lloyd_avx(data, parameters);
				verify_clustering_results($, serial_res, avx_res);
			}
		}
	},

	CASE("Verify correctness with benchmark dataset: dim128.data.csv",) {
		SETUP(lest::env& $) {
			auto data = dkm::load_csv<float, 128>("../bench/dim128.data.csv");
			dkm::clustering_parameters<float> parameters(16);
			parameters.set_random_seed(42);

			auto serial_res = dkm::kmeans_lloyd(data, parameters);
			
			SECTION("Parallel version matches serial version") {
				auto parallel_res = dkm::kmeans_lloyd_parallel(data, parameters);
				verify_clustering_results($, serial_res, parallel_res);
			}

			SECTION("AVX version matches serial version") {
				auto avx_res = dkm::kmeans_lloyd_avx(data, parameters);
				verify_clustering_results($, serial_res, avx_res);
			}
		}
	},

	CASE("Test with real data set",) {
		SETUP() {
			auto data = dkm::load_csv<float, 2>("iris.data.csv");
			dkm::clustering_parameters<float> parameters(3);
			parameters.set_random_seed(random_seed_value);

			SECTION("Segmentation completes to convergence") {
				auto means_clusters = dkm::kmeans_lloyd(data, parameters);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);

				EXPECT(means.size() == 3u);
				EXPECT(clusters.size() == data.size());
				std::vector<std::array<float, 2>> expected_means {
					{3.44082f, 0.242857f},
					{2.70755f, 1.30943f},
					{3.04167f, 2.05208f},
				};
				for (size_t i = 0; i < means.size(); ++i) {
					for (size_t j = 0; j < means[i].size(); ++j) {
						EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
					}
				}
				// not checking clusters here because there are too many points
			}

			SECTION("Segmentation completes to convergence with parallel implementation") {
				auto means_clusters = dkm::kmeans_lloyd_parallel(data, parameters);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);

				EXPECT(means.size() == 3u);
				EXPECT(clusters.size() == data.size());
				std::vector<std::array<float, 2>> expected_means {
					{3.44082f, 0.242857f},
					{2.70755f, 1.30943f},
					{3.04167f, 2.05208f},
				};
				for (size_t i = 0; i < means.size(); ++i) {
					for (size_t j = 0; j < means[i].size(); ++j) {
						EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
					}
				}
				// not checking clusters here because there are too many points
			}

			SECTION("Segmentation completes early because iteration limit is reached") {
				parameters.set_max_iteration(5);
				auto means_clusters = dkm::kmeans_lloyd(data, parameters);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);

				EXPECT(means.size() == 3u);
				EXPECT(clusters.size() == data.size());
				std::vector<std::array<float, 2>> expected_means {
					{3.418, 0.244f},
					{2.72857, 1.41587f},
					{3.11622f, 2.11892f},
				};
				for (size_t i = 0; i < means.size(); ++i) {
					for (size_t j = 0; j < means[i].size(); ++j) {
						EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
					}
				}
				// not checking clusters here because there are too many points
			}

			SECTION("Segmentation completes early because iteration limit is reached with parallel implementation") {
				parameters.set_max_iteration(5);
				auto means_clusters = dkm::kmeans_lloyd_parallel(data, parameters);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);

				EXPECT(means.size() == 3u);
				EXPECT(clusters.size() == data.size());
				std::vector<std::array<float, 2>> expected_means {
					{3.418, 0.244f},
					{2.72857, 1.41587f},
					{3.11622f, 2.11892f},
				};
				for (size_t i = 0; i < means.size(); ++i) {
					for (size_t j = 0; j < means[i].size(); ++j) {
						EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
					}
				}
				// not checking clusters here because there are too many points
			}
		}
	},

	CASE("Test dkm::get_cluster",) {
		SETUP() {
			std::vector<std::array<double, 2>> points{
				{0, 0},
				{1, 1},
				{2, 2},
				{3, 3},
				{4, 4},
				{5, 5},
				{6, 6},
				{7, 7},
				{8, 8},
				{9, 9},
			};
			std::vector<uint32_t> labels{0, 2, 1, 1, 0, 2, 2, 1, 1, 0};
			SECTION("Non-empty and same size points and labels") {

				SECTION("Correct points for existing labels") {
					auto cluster = dkm::get_cluster(points, labels, 0);
					std::vector<std::array<double, 2>> res{
						{0, 0},
							{4, 4},
							{9, 9}
					};
					EXPECT(cluster == res);

					cluster = dkm::get_cluster(points, labels, 1);
					res = {
						{2, 2},
						{3, 3},
						{7, 7},
						{8, 8}
					};
					EXPECT(cluster == res);

					cluster = dkm::get_cluster(points, labels, 2);
					res = {
						{1, 1},
						{5, 5},
						{6, 6},
					};
					EXPECT(cluster == res);
				}

				SECTION("Empty set of points for non-existing labels") {
					auto cluster = dkm::get_cluster(points, labels, 4);
					std::vector<std::array<double, 2>> empty;
					EXPECT(cluster == empty);
				}
			}

			SECTION("Empty points and labels") {
				std::vector<std::array<double, 2>> points;
				std::vector<uint32_t> labels;

				SECTION("Empty set of points") {
					auto cluster = dkm::get_cluster(points, labels, 0);
					std::vector<std::array<double, 2>> empty;
					EXPECT(cluster == empty);
				}
			}

			SECTION("points and labels sequences with different sizes") {
				std::vector<std::array<double, 2>> points{
					{0, 1},
					{2, 3.5}
				};
				std::vector<uint32_t> labels{2, 4, 1, 1};
			}
		}
	},

	CASE("Test dkm::dist_to_center",) {
		SETUP() {
			std::vector<std::array<double, 2>> points{
				{1, 5},
				{2.2, 3},
				{8, 12},
				{11.4, 4.87},
				{0.27, 50},
				{1, 1}
			};
			std::array<double, 2> center{17.2, 24.5};

			std::vector<double> res{25.3513, 26.2154, 15.5206, 20.4689, 30.6084, 28.5427};
			SECTION("Non-empty sequence of points") {

				std::vector<double> out = dkm::dist_to_center(points, center);

				for (size_t i = 0; i < out.size(); ++i)
					EXPECT(lest::approx(out[i]) == res[i]);
			}

			SECTION("Empty sequence of points returns an empty vector") {
				std::vector<std::array<double, 2>> points;
				std::array<double, 2> center{5, 4};

				std::vector<double> empty;
				std::vector<double> out = dkm::dist_to_center(points, center);

				EXPECT(out == empty);
			}
		}
	},

	CASE("Test dkm::sum_dist",) {
		SETUP() {
			std::vector<std::array<double, 2>> points{
				{1,    5},
				{2.2,  3},
				{8,    12},
				{11.4, 4.87},
				{0.27, 50},
				{1,    1}
			};
			std::vector<double> out(points.size());
			std::array<double, 2> center{17.2, 24.5};

			std::vector<double> res{25.3513, 26.2154, 15.5206, 20.4689, 30.6084, 28.5427};
			SECTION("Non-empty sequence of points") {

				EXPECT(dkm::sum_dist(points, center) == lest::approx(146.7073));
			}

			SECTION("Empty sequence of points returns 0") {
				std::vector<std::array<double, 2>> points;
				std::array<double, 2> center{5, 4};

				EXPECT(dkm::sum_dist(points, center) == 0);
			}
		}
	},

	CASE("Test dkm::means_inertia",) {
		SETUP() {
			std::vector<std::array<double, 2>> points{
				{66.01742226,  48.70477854},
				{62.30094932, 108.44049522},
				{39.60740312,  12.07668535},
				{35.57096194,  -7.10722525},
				{39.90890238,  61.89509695},
				{27.5850295 ,  85.50226002},
				{51.14012591,  27.90650051},
				{58.6414776 ,  31.97020798},
				{14.75127435,  69.36707669},
				{73.66255253,  84.73455103},
				{-1.31034384,  66.10406579},
				{41.91865987,  56.5003107 },
				{33.31116528,  45.92203855},
				{57.12362692,  37.73753163},
				{ 2.68915431,  51.35514789},
				{39.76543196,  -5.99499795},
				{72.64312341,  61.43756623},
				{30.97140948,  29.49960625},
				{25.31232669,  35.88059477},
				{57.67046396,  35.05019015}
			};
			std::vector<std::array<double, 2>> centroids{
				{10, 10},
				{20, 20},
				{40, 30}
			};
			std::vector<uint32_t> labels{
				0, 0, 1, 2, 2, 1, 1, 0, 0, 0,
				1, 1, 2, 1, 0, 0, 1, 2, 1, 0
			};
			uint32_t k = 3;
			SECTION("Non-empty set of points, fixed 3 clusters") {
				std::tuple<std::vector<std::array<double, 2>>, std::vector<uint32_t>> means{centroids, labels};

				double inertia = 0;
				for (size_t i = 0; i < labels.size(); ++i) {
					auto center = centroids[labels[i]];
					auto point = points[i];
					inertia += dkm::details::distance(point, center);
				}

				EXPECT(lest::approx(inertia) == dkm::means_inertia(points, means, k));
			}

			SECTION("Empty set of points should give 0 inertia") {
				std::vector<std::array<double, 2>> points;
				std::tuple<std::vector<std::array<double, 2>>, std::vector<uint32_t>> means;

				EXPECT(dkm::means_inertia(points, means, k) == lest::approx(0));
			}

			SECTION() {
				std::vector<std::array<double, 2>> data{
					{1, 1},
						{2, 2},
						{1200, 1200},
						{1000, 1000}
				};
				uint32_t k = 2;
				auto means = dkm::kmeans_lloyd(data, k);
				double inertia = dkm::means_inertia(data, means, k);
				EXPECT(284.256926 == lest::approx(inertia).epsilon(1e-6));
			}
		}
	},
	CASE("Test dkm::get_best_means",) {
		SETUP() {
			std::vector<std::array<double, 2>> points{
				{8,  8},
				{9, 9},
				{11,  11},
				{12,  12},
				{18,  18},
				{19,  19},
				{21,  21},
				{22,  22},
				{39,  39},
				{41,  41},
			};
			std::vector<std::array<double, 2>> centroids{
				{10, 10},
				{20, 20},
				{40, 40}
			};
			std::vector<uint32_t> labels{0, 0, 0, 0, 1, 1, 1, 1, 2, 2};
			uint32_t k = 3;
			SECTION("Test if we get the clustering with the least inertia") {
				auto means = dkm::get_best_means(points, k, 20);
				std::vector<std::array<double, 2>> returned_centroids;
				std::vector<uint32_t> returned_labels;
				std::tie(returned_centroids, returned_labels) = means;
				// every point is assigned to the same cluster center
				for (uint32_t i = 0; i < points.size(); ++i) {
					auto expected_center = centroids[labels[i]];
					auto returned_center = returned_centroids[returned_labels[i]];
					EXPECT(expected_center[0] == lest::approx(returned_center[0]));
					EXPECT(expected_center[1] == lest::approx(returned_center[1]));
				}
			}
		}
	},
	CASE("Test dkm::predict",) {
		SETUP() {
			std::vector<std::array<double, 2>> centroids{
					{8,  8},
					{9, 9},
					{11,  11},
					{12,  12},
					{18,  18},
					{19,  19},
					{21,  21},
					{22,  22},
					{39,  39},
					{41,  41},
			};
			std::array<double, 2> query {11, 10.5};
			SECTION("Test if we get the actual closest centroid to the query") {
				auto res = dkm::predict(centroids, query);
				EXPECT(res == 2u);
			}
		}
	}
};

int main(int argc, char** argv) {
	std::cout << "Starting tests..." << std::endl;
	int result = lest::run(specification, argc, argv);
	std::cout << "Tests finished with code: " << result << std::endl;
	return result;
}

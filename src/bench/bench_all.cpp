#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// Include all dkm headers to be benchmarked
#include "../../include/dkm.hpp"
#include "../../include/dkm_utils.hpp"
// ... all other dkm headers ...
#include "../../include/dkm_pt_avx.hpp"
#include "../../include/dkm_pt_avx_prealign.hpp"
#include "../../include/dkm_pt_avx_unalign.hpp"
#include "../../include/dkm_pthread.hpp"
#include "../../include/dkm_thread_pool.hpp"
#include "../../include/dkm_thread_pool_v2.hpp"
#include "../../include/dkm_tp_avx.hpp"
#include "../../include/dkm_tp_avx_prealign.hpp"
#include "../../include/dkm_tp_avx_unalign.hpp"
#include "../../include/dkm_tp_v2_avx.hpp"
#include "../../include/dkm_tp_v2_avx_prealign.hpp"
#include "../../include/dkm_tp_v2_avx_unalign.hpp"

constexpr uint64_t random_seed_value = 7;
const int BENCH_ITERATIONS = 100;

// Generic profiling function
template <typename Func, typename T, size_t N>
double profile_dkm_generic(
	Func kmeans_func, const std::vector<std::array<T, N>>& data, const dkm::clustering_parameters<T>& params) {
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < BENCH_ITERATIONS; ++i) {
		auto result = kmeans_func(data, params);
		(void)result;
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> total_time = end - start;
	return total_time.count() / BENCH_ITERATIONS;
}

using WideBenchResult = std::map<std::string, std::map<uint32_t, double>>;

void write_wide_csv(const std::string& dataset_name,
	const WideBenchResult& results,
	const std::vector<uint32_t>& thread_counts,
	const std::vector<std::string>& method_order) {

	std::string filename = "results_" + dataset_name;
	size_t pos = filename.find(".data.csv");
	if (pos != std::string::npos) {
		filename.replace(pos, 9, ".csv");
	}

	std::ofstream file(filename);
	if (!file.is_open()) {
		return;
	}

	// Header
	file << "Method";
	if (std::find(thread_counts.begin(), thread_counts.end(), 1) != thread_counts.end() || results.count("Serial")) {
		file << ",1 Thread";
	}
	for (uint32_t t : thread_counts) {
		if (t > 1) {
			file << "," << t << " Threads";
		}
	}
	file << "\n";

	for (const std::string& method : method_order) {
		// 檢查方法是否有測試結果
		if (results.find(method) == results.end()) {
			continue;
		}

		const auto& thread_times = results.at(method);

		file << method;

		if (std::find(thread_counts.begin(), thread_counts.end(), 1) != thread_counts.end() || method == "Serial") {
			if (thread_times.count(1)) {
				file << "," << std::fixed << std::setprecision(4) << thread_times.at(1);
			} else {
				file << ",";
			}
		}

		for (uint32_t t : thread_counts) {
			if (t > 1) {
				if (thread_times.count(t)) {
					file << "," << std::fixed << std::setprecision(4) << thread_times.at(t);
				} else {
					file << ",";
				}
			}
		}
		file << "\n";
	}

	std::cout << "Benchmark results for " << dataset_name << " written to " << filename << std::endl;
}

template <typename T, size_t N>
void run_all_benchmarks_for_dataset(const std::string& path,
	uint32_t k,
	const std::vector<uint32_t>& thread_counts,
	const std::vector<std::string>& method_order) {
	std::cout << "\n## Benchmarking Dataset: " << path << " (K=" << k << ", N=" << N << ") ##" << std::endl;
	auto data = dkm::load_csv<T, N>(path);
	if (data.empty()) {
		std::cerr << "Warning: Could not load data from " << path << std::endl;
		return;
	}

	WideBenchResult wide_results;

	using kmeans_func_t = std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> (*)(
		const std::vector<std::array<T, N>>&, const dkm::clustering_parameters<T>&);

	// --- Serial version ---
	std::cout << "Running Serial..." << std::endl;
	dkm::clustering_parameters<T> serial_params(k);
	serial_params.set_random_seed(random_seed_value);
	auto serial_func_ptr = static_cast<kmeans_func_t>(dkm::kmeans_lloyd<T, N>);
	double serial_time = profile_dkm_generic(serial_func_ptr, data, serial_params);
	wide_results["Serial"][1] = serial_time;


	// --- Parallel versions ---
	for (uint32_t t : thread_counts) {
		std::cout << "Running for " << t << " threads..." << std::endl;
		dkm::clustering_parameters<T> params(k);
		params.set_random_seed(random_seed_value);
		params.set_num_threads(t);

		auto run = [&](const std::string& name, kmeans_func_t func) {
			double time_ms = profile_dkm_generic(func, data, params);
			wide_results[name][t] = time_ms;
		};

		// Pthread
		run("Pthread", static_cast<kmeans_func_t>(dkm::kmeans_lloyd_pt<T, N>));
		run("Pthread_AVX", static_cast<kmeans_func_t>(dkm::kmeans_lloyd_pt_avx<T, N>));
		run("Pthread_AVX_Unalign", static_cast<kmeans_func_t>(dkm::kmeans_lloyd_pt_avx_unalign<T, N>));
		run("Pthread_AVX_Prealign", static_cast<kmeans_func_t>(dkm::kmeans_lloyd_pt_avx_prealign<T, N>));

		// ThreadPool V1
		run("Thread_Pool", static_cast<kmeans_func_t>(dkm::kmeans_lloyd_tp<T, N>));
		run("Thread_Pool_AVX", static_cast<kmeans_func_t>(dkm::kmeans_lloyd_tp_avx<T, N>));
		run("Thread_Pool_AVX_Unalign", static_cast<kmeans_func_t>(dkm::kmeans_lloyd_tp_avx_unalign<T, N>));
		run("Thread_Pool_AVX_Prealign", static_cast<kmeans_func_t>(dkm::kmeans_lloyd_tp_avx_prealign<T, N>));

		// ThreadPool V2
		run("Thread_Pool_V2", static_cast<kmeans_func_t>(dkm::kmeans_lloyd_tp_v2<T, N>));
		run("Thread_Pool_V2_AVX", static_cast<kmeans_func_t>(dkm::kmeans_lloyd_tp_v2_avx<T, N>));
		run("Thread_Pool_V2_AVX_Unalign", static_cast<kmeans_func_t>(dkm::kmeans_lloyd_tp_v2_avx_unalign<T, N>));
		run("Thread_Pool_V2_AVX_Prealign", static_cast<kmeans_func_t>(dkm::kmeans_lloyd_tp_v2_avx_prealign<T, N>));
	}

	write_wide_csv(path, wide_results, thread_counts, method_order);
}


int main() {
	std::cout << "# BEGINNING BENCHMARKS #\n" << std::endl;

	const std::vector<uint32_t> thread_counts = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<std::string> method_order = {
		"Serial",
		"Pthread",
		"Pthread_AVX",
		"Pthread_AVX_Unalign",
		"Pthread_AVX_Prealign",
		"Thread_Pool",
		"Thread_Pool_AVX",
		"Thread_Pool_AVX_Unalign",
		"Thread_Pool_AVX_Prealign",
		"Thread_Pool_V2",
		"Thread_Pool_V2_AVX",
		"Thread_Pool_V2_AVX_Unalign",
		"Thread_Pool_V2_AVX_Prealign"
	};

	run_all_benchmarks_for_dataset<float, 2>("iris.data.csv", 3, thread_counts, method_order);
	run_all_benchmarks_for_dataset<float, 2>("s1.data.csv", 15, thread_counts, method_order);
	run_all_benchmarks_for_dataset<float, 2>("birch3.data.csv", 100, thread_counts, method_order);
	run_all_benchmarks_for_dataset<float, 128>("dim128.data.csv", 16, thread_counts, method_order);

	std::cout << "\n# ALL BENCHMARKS FINISHED #\n" << std::endl;

	return 0;
}
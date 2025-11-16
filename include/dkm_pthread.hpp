#pragma once
#ifndef DKM_PTHREAD_KMEANS_H
#define DKM_PTHREAD_KMEANS_H

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
#include "dkm.hpp"   

namespace dkm {

namespace details {

template <typename T, size_t N>
struct ThreadArgs {
    size_t id;
    size_t begin;
    size_t end;
    const std::vector<std::array<T, N>>* data;
    const std::vector<std::array<T, N>>* means;
    std::vector<T>* distances;         // for closest_distance_pt
    std::vector<uint32_t>* clusters;   // for calculate_clusters_pt
};

template <typename T>
inline size_t determine_num_threads(const clustering_parameters<T>& parameters) {
    if (parameters.has_num_threads()) {
        return parameters.get_num_threads();
    } else {
        return std::thread::hardware_concurrency();
    }
}

/*
pthread worker：計算每個點到最近中心的歐式距離平方
*/
template <typename T, size_t N>
void* worker_closest_distance(void* arg) {
    auto* a = static_cast<details::ThreadArgs<T, N>*>(arg);
    const auto& data  = *a->data;
    const auto& means = *a->means;
    auto& distances   = *a->distances;

    const size_t M = means.size();
    for (size_t i = a->begin; i < a->end; ++i) {
        T best = details::distance_squared<T, N>(data[i], means[0]);
        for (size_t m = 1; m < M; ++m) {
            const T d2 = details::distance_squared<T, N>(data[i], means[m]);
            if (d2 < best) best = d2;
        }
        distances[i] = best;
    }
    return nullptr;
}

template <typename T, size_t N>
std::vector<T> closest_distance_pt(
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
        pthread_create(&threads[t], nullptr, &details::worker_closest_distance<T, N>, &a);
    }
    for (size_t t = 0; t < nthr; ++t) pthread_join(threads[t], nullptr);
    return distances;
}

/*
This is an alternate initialization method based on the [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B)
initialization algorithm.
*/
template <typename T, size_t N>
std::vector<std::array<T, N>> random_plusplus_pt(
	const std::vector<std::array<T, N>>& data, uint32_t k, uint64_t seed, const clustering_parameters<T>& parameters) 
{
	assert(k > 0);
	assert(data.size() > 0);
	using input_size_t = typename std::array<T, N>::size_type;
	std::vector<std::array<T, N>> means;

	// Using a very simple PRBS generator, parameters selected according to
	// https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
	std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX> rand_engine(seed);

	// Select first mean at random from the set
	{
		std::uniform_int_distribution<input_size_t> uniform_generator(0, data.size() - 1);
		means.push_back(data[uniform_generator(rand_engine)]);
	}

	for (uint32_t count = 1; count < k; ++count) {
		// Calculate the distance to the closest mean for each data point
		auto distances = details::closest_distance_pt(means, data, parameters);
		
		// Pick a random point weighted by the distance from existing means
		// TODO: This might convert floating point weights to ints, distorting the distribution for small weights
#if !defined(_MSC_VER) || _MSC_VER >= 1900
		std::discrete_distribution<input_size_t> generator(distances.begin(), distances.end());
#else  // MSVC++ older than 14.0
		input_size_t i = 0;
		std::discrete_distribution<input_size_t> generator(distances.size(), 0.0, 0.0, [&distances, &i](double) { return distances[i++]; });
#endif
		means.push_back(data[generator(rand_engine)]);
	}
	return means;
}

/*
pthread worker：計算每個點最近的中心索引
*/
template <typename T, size_t N>
void* worker_closest_mean(void* arg) {
    auto* a = static_cast<details::ThreadArgs<T, N>*>(arg);
    const auto& data  = *a->data;
    const auto& means = *a->means;
    auto& clusters    = *a->clusters;

    const size_t M = means.size();
    for (size_t idx = a->begin; idx < a->end; ++idx) {
        T best = details::distance_squared<T, N>(data[idx], means[0]);
        uint32_t best_id = 0;
        for (size_t m = 1; m < M; ++m) {
            const T d2 = details::distance_squared<T, N>(data[idx], means[m]);
            if (d2 < best) { best = d2; best_id = static_cast<uint32_t>(m); }
        }
        clusters[idx] = best_id;
    }
    return nullptr;
}

/*
平行版：計算每個資料點的最近群中心
*/
template <typename T, size_t N>
std::vector<uint32_t> calculate_clusters_pt(
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
        pthread_create(&threads[t], nullptr, &details::worker_closest_mean<T, N>, &a);
    }
    for (size_t t = 0; t < nthr; ++t) pthread_join(threads[t], nullptr);
    return clusters;
}


} // namespace details


template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>kmeans_lloyd_pt(
    const std::vector<std::array<T, N>>& data,
    const clustering_parameters<T>& parameters)
{
    static_assert(std::is_arithmetic<T>::value && std::is_signed<T>::value,
        "kmeans_lloyd requires the template parameter T to be a signed arithmetic type (e.g. float, double, int)");
    assert(parameters.get_k() > 0);  // k must be greater than zero
    assert(data.size() >= parameters.get_k());  // there must be at least k data points
    std::random_device rand_device;
    uint64_t seed = parameters.has_random_seed() ? parameters.get_random_seed() : rand_device();
    std::vector<std::array<T, N>> means = details::random_plusplus_pt<T, N>(data, parameters.get_k(), seed, parameters);

    std::vector<std::array<T, N>> old_means;
    std::vector<std::array<T, N>> old_old_means;
    std::vector<uint32_t> clusters;
    
    // Calculate new means until convergence is reached or we hit the maximum iteration count
    uint64_t count = 0;
    do {
        clusters = details::calculate_clusters_pt<T, N>(data, means, parameters);
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
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd_pt(
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
    return kmeans_lloyd_pt<T, N>(data, parameters);
}

} // namespace dkm

#endif /* DKM_PTHREAD_KMEANS_H */

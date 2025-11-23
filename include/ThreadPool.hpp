#pragma once

#include <pthread.h>
#include <queue>
#include <vector>
#include <functional>
#include <mutex>
#include <condition_variable>

// 綁定 CPU 用
#include <unistd.h>  // sysconf
#include <sched.h>   // CPU_SET / pthread_setaffinity_np

class ThreadPool {
public:
    ThreadPool() : ThreadPool(sysconf(_SC_NPROCESSORS_ONLN) > 0 ? sysconf(_SC_NPROCESSORS_ONLN) : 1) {}

    explicit ThreadPool(size_t nthreads) : stop(false), active_tasks(0) {
        workers.resize(nthreads);
        for (size_t i = 0; i < nthreads; ++i) {
            pthread_create(&workers[i], nullptr, ThreadPool::worker_entry, this);
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mtx);
            stop = true;
            cv_task.notify_all();
        }
        for (auto& t : workers) pthread_join(t, nullptr);
    }

    void reset(size_t new_nthreads) {
        if (new_nthreads == 0) new_nthreads = 1;

        if (new_nthreads == workers.size() && !stop) {
            return;
        }

        // Cleanly shut down existing workers
        {
            std::unique_lock<std::mutex> lock(mtx);
            stop = true;
            cv_task.notify_all();
        }
        for (auto& t : workers) {
            pthread_join(t, nullptr);
        }

        // Re-initialize with the new number of threads
        workers.clear();
        workers.resize(new_nthreads);

        std::queue<std::function<void()>> empty_queue;
        tasks.swap(empty_queue);

        stop = false;
        active_tasks = 0;

        for (size_t i = 0; i < new_nthreads; ++i) {
            pthread_create(&workers[i], nullptr, ThreadPool::worker_entry, this);
        }
    }

    // 放入任務
    void enqueue(std::function<void()> fn) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            tasks.push(std::move(fn));
            ++active_tasks;
        }
        cv_task.notify_one();
    }

    // 等所有任務完成
    void wait_all() {
        std::unique_lock<std::mutex> lock(mtx);
        cv_done.wait(lock, [&]() {
            return active_tasks == 0 && tasks.empty();
        });
    }

private:
    std::vector<pthread_t> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex mtx;
    std::condition_variable cv_task;
    std::condition_variable cv_done;

    bool stop;
    size_t active_tasks;

    static void* worker_entry(void* arg) {
        static_cast<ThreadPool*>(arg)->worker_loop();
        return nullptr;
    }

    void worker_loop() {
        
        // 進來先把自己綁到某顆 core
        {
            int cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
            if (cpu_count > 0) {
                pthread_t self = pthread_self();

                // 找出自己在 workers 裡的 index
                size_t idx = 0;
                for (; idx < workers.size(); ++idx) {
                    if (pthread_equal(workers[idx], self)) {
                        break;
                    }
                }

                if (idx < workers.size()) {
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    // thread i 綁到 core (i % cpu_count)
                    CPU_SET(static_cast<int>(idx % cpu_count), &cpuset);

                    // 忽略錯誤處理，失敗就沒綁上 CPU
                    pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset);
                }
            }
        }

        while (true) {
            std::function<void()> fn;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_task.wait(lock, [&]() {
                    return stop || !tasks.empty();
                });
                if (stop && tasks.empty()) return;

                fn = std::move(tasks.front());
                tasks.pop();
            }

            fn(); // execute

            {
                std::unique_lock<std::mutex> lock(mtx);
                --active_tasks;
                if (active_tasks == 0 && tasks.empty())
                    cv_done.notify_all();
            }
        }
    }
};

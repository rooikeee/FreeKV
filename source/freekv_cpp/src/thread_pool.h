#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <stdexcept>
#include <memory> // For unique_ptr
#include <atomic> // For atomic_bool

#include <thread>
#include <future>
#include <utility>
#include <iostream>

class ThreadPool {
public:
  ThreadPool(size_t threads) : stop(false) {
    for(size_t i = 0; i<threads; ++i)
      workers.emplace_back([this] {
        for(;;) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(lock,
                [this]{ return this->stop || !this->tasks.empty(); });
            if(this->stop && this->tasks.empty())
                return;
            task = std::move(this->tasks.front());
            this->tasks.pop();
          }
          try {
              task();
          } catch (const std::exception& e) {
              std::cerr << "Exception caught in thread pool task: " << e.what() << std::endl;
          } catch (...) {
              std::cerr << "Unknown exception caught in thread pool task." << std::endl;
          }
        }
      });
    }

  template<class F, class... Args>
  // Use std::function to avoid template in enqueue signature for easier binding
  void enqueue(F&& f, Args&&... args) {
    auto task = std::make_shared< std::packaged_task<void()> >(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");
      tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    // Note: We are not returning the future here as Python side will use CUDA events
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
      worker.join();
  }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop; // Use atomic for thread safety
};

extern std::unique_ptr<ThreadPool> recall_thread_pool;
extern std::once_flag pool_init_flag;

void init_recall_thread_pool(int num_threads);

void shutdown_recall_thread_pool();

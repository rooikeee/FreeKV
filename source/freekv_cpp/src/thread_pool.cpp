#include "thread_pool.h"
#include <thread>

std::unique_ptr<ThreadPool> recall_thread_pool;
std::once_flag pool_init_flag;

void init_recall_thread_pool(int num_threads) {
  std::call_once(pool_init_flag, [&]() {
    recall_thread_pool = std::make_unique<ThreadPool>(
      num_threads > 0 ? num_threads : std::thread::hardware_concurrency()
    );
  });
}

void shutdown_recall_thread_pool() {
  if (recall_thread_pool) {
    recall_thread_pool.reset(); // Destructor joins threads
  }
}
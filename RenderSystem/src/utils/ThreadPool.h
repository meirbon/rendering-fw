#pragma once

#include <thread>
#include <mutex>
#include <queue>
#include <future>
#include <atomic>
#include <functional>

namespace rfw
{
namespace utils
{
class ThreadPool
{
  private:
	template <typename T> class Queue
	{
	  public:
		bool push(const T &val)
		{
			auto lock = std::lock_guard(m_Mutex);
			m_Queue.push(val);
			return true;
		}

		bool pop(T &front)
		{
			auto lock = std::lock_guard(m_Mutex);
			if (m_Queue.empty())
				return false;
			front = m_Queue.front();
			m_Queue.pop();
			return true;
		}

		bool empty()
		{
			auto lock = std::lock_guard(m_Mutex);
			return m_Queue.empty();
		}

	  private:
		std::queue<T> m_Queue;
		std::mutex m_Mutex;
	};

  public:
	ThreadPool() { init(); }
	ThreadPool(size_t threadCount)
	{
		init();
		resize(threadCount);
	}

	size_t size() const { return m_Threads.size(); }
	size_t idleCount() const { return m_WaitingThreadCount; }
	std::thread &getThread(size_t ID) { return *m_Threads[ID]; }

	void resize(size_t threadCount)
	{
		if (!m_Stop && !m_Done)
		{
			const auto oldThreadCount = m_Threads.size();
			if (oldThreadCount <= threadCount)
			{
				m_Threads.resize(threadCount);
				m_Flags.resize(threadCount);

				for (size_t i = oldThreadCount; i < threadCount; i++)
				{
					m_Flags[i] = std::make_shared<std::atomic_bool>(false);
					setThread(i);
				}
			}
			else
			{
				for (size_t i = oldThreadCount - 1; i >= threadCount; i--)
				{
					// Force thread to finish
					(*m_Flags[i]) = true;
					m_Threads[i]->detach();
				}
				{
					std::unique_lock<std::mutex> lock(m_Mutex);
					m_CV.notify_all();
				}

				m_Threads.resize(threadCount);
				m_Flags.resize(threadCount);
			}
		}
	}

	void clearQueue()
	{
		std::function<void(int ID)> *callback;
		while (m_Queue.pop(callback))
			delete callback;
	}

	std::function<void(int)> pop()
	{
		std::function<void(int ID)> *callback = nullptr;
		m_Queue.pop(callback);
		std::unique_ptr<std::function<void(int ID)>> func(callback);
		std::function<void(int)> cb;
		if (callback)
			cb = *callback;
		return cb;
	}

	void stop(bool wait = false)
	{
		if (wait)
		{
			if (m_Stop)
				return;

			for (int i = 0, n = static_cast<int>(size()); i < n; i++)
				(*m_Flags[i]) = true;

			clearQueue();
		}
		else
		{
			if (m_Done || m_Stop)
				return;
			m_Done = true;
		}

		{
			std::unique_lock<std::mutex> lock(m_Mutex);
			m_CV.notify_all();
		}

		for (int i = 0, s = static_cast<int>(m_Threads.size()); i < s; i++)
		{
			if (m_Threads[i]->joinable())
				m_Threads[i]->join();
		}

		clearQueue();
		m_Threads.clear();
		m_Flags.clear();
	}

	template <typename T, typename... Rest> auto push(T &&f, Rest &&... rest) -> std::future<decltype(f(0, rest...))>
	{
		auto pck = std::make_shared<std::packaged_task<decltype(f(0, rest...))(int)>>(
			std::bind(std::forward<T>(f), std::placeholders::_1, std::forward<Rest>(rest)...));

		auto cb = new std::function<void(int ID)>([pck](int ID) { (*pck)(ID); });

		m_Queue.push(cb);
		std::unique_lock<std::mutex> lock(m_Mutex);
		m_CV.notify_one();
		return pck->get_future();
	}

	template <typename T> auto push(T &&t) -> std::future<decltype(t(0))>
	{
		auto pck = std::make_shared<std::packaged_task<decltype(t(0))(int)>>(std::forward<T>(t));
		auto cb = new std::function<void(int id)>([pck](int id) { (*pck)(id); });

		m_Queue.push(cb);
		std::unique_lock<std::mutex> lock(m_Mutex);
		m_CV.notify_one();
		return pck->get_future();
	}

  private:
	// Prevent threadpools from being moved around
	ThreadPool(const ThreadPool &other) = delete;
	ThreadPool(ThreadPool &&) = delete;
	ThreadPool &operator=(const ThreadPool &);
	ThreadPool &operator=(ThreadPool &&);

	void setThread(size_t ID)
	{
		// Copy of the shared ptr to the flag
		std::shared_ptr<std::atomic<bool>> flag(m_Flags[ID]);
		auto f = [this, ID, flag]() {
			std::atomic<bool> &_flag = *flag;
			std::function<void(int id)> *callback;
			bool isPop = m_Queue.pop(callback);
			while (true)
			{
				while (isPop)
				{																 // if there is anything in the queue
					std::unique_ptr<std::function<void(int id)>> func(callback); // at return, delete the function even if an exception occurred
					(*callback)(ID);
					if (_flag)
						return; // the thread is wanted to stop, return even if the queue is not empty yet
					else
						isPop = m_Queue.pop(callback);
				}
				// the queue is empty here, wait for the next command
				std::unique_lock<std::mutex> lock(m_Mutex);
				++m_WaitingThreadCount;
				m_CV.wait(lock, [this, &callback, &isPop, &_flag]() {
					isPop = m_Queue.pop(callback);
					return isPop || m_Done || _flag;
				});

				// if the queue is empty and this->isDone == true or *flag then return
				--m_WaitingThreadCount;
				if (!isPop)
					return;
			}
		};

		// compiler may not support std::make_unique()
		m_Threads[ID].reset(new std::thread(f));
	}

	void init()
	{
		m_WaitingThreadCount = 0;
		m_Done = false;
		m_Stop = false;
	}

  private:
	std::vector<std::unique_ptr<std::thread>> m_Threads;
	std::vector<std::shared_ptr<std::atomic_bool>> m_Flags;
	Queue<std::function<void(int tID)> *> m_Queue;

	std::atomic_bool m_Done;
	std::atomic_bool m_Stop;
	std::atomic_int m_WaitingThreadCount;

	std::mutex m_Mutex;
	std::condition_variable m_CV;
};
} // namespace utils
} // namespace rfw
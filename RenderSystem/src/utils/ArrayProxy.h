#pragma once

#include <algorithm>
#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>

namespace rfw::utils
{
template <typename T> class ArrayProxy
{
  public:
	constexpr ArrayProxy(std::nullptr_t) : m_Size(0), m_Data(nullptr) {}

	ArrayProxy(T &ptr) : m_Size(1), m_Data(&ptr) {}

	ArrayProxy(uint32_t count, T *ptr) : m_Size(count), m_Data(ptr) {}

	template <size_t N>
	ArrayProxy(std::array<typename std::remove_const<T>::type, N> &data) : m_Size(N), m_Data(data.data())
	{
	}

	template <size_t N>
	ArrayProxy(std::array<typename std::remove_const<T>::type, N> const &data) : m_Size(N), m_Data(data.data())
	{
	}

	template <class Allocator = std::allocator<typename std::remove_const<T>::type>>
	ArrayProxy(std::vector<typename std::remove_const<T>::type, Allocator> &data)
		: m_Size(static_cast<uint32_t>(data.size())), m_Data(data.data())
	{
	}

	template <class Allocator = std::allocator<typename std::remove_const<T>::type>>
	ArrayProxy(std::vector<typename std::remove_const<T>::type, Allocator> const &data)
		: m_Size(static_cast<uint32_t>(data.size())), m_Data(data.data())
	{
	}

	ArrayProxy(std::initializer_list<T> const &data)
		: m_Size(static_cast<uint32_t>(data.end() - data.begin()))
	{
		m_Data = data.begin();
	}

	const T *data() const { return m_Data; }

	const T &at(size_t index) const { return m_Data[index]; }

	const T *begin() const { return m_Data; }

	const T *end() const { return m_Data + m_Size; }

	const T &front() const
	{
		VULKAN_HPP_ASSERT(m_Size && m_Data);
		return *m_Data;
	}

	const T &back() const
	{
		VULKAN_HPP_ASSERT(m_Size && m_Data);
		return *(m_Data + m_Size - 1);
	}

	bool empty() const { return (m_Size == 0); }

	size_t size() const { return m_Size; }

  private:
	size_t m_Size;
	const T *m_Data;
};
} // namespace rfw::utils
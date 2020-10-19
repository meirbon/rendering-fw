#pragma once

#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <system_error>
#include <type_traits>
#include <cassert>

namespace rfw::utils
{
template <typename T> class array_proxy
{
  public:
	constexpr array_proxy(std::nullptr_t) : m_Size(0), m_Data(nullptr) {}

	array_proxy(T &data) : m_Size(1), m_Data(&data) {}

	array_proxy(uint32_t count, T *ptr) : m_Size(count), m_Data(ptr) {}

	template <size_t N>
	array_proxy(std::array<typename std::remove_const<T>::type, N> &data) : m_Size(N), m_Data(data.data())
	{
	}

	template <size_t N>
	array_proxy(std::array<typename std::remove_const<T>::type, N> const &data) : m_Size(N), m_Data(data.data())
	{
	}

	template <class Allocator = std::allocator<typename std::remove_const<T>::type>>
	array_proxy(std::vector<typename std::remove_const<T>::type, Allocator> &data)
		: m_Size(static_cast<uint32_t>(data.size())), m_Data(data.data())
	{
	}

	template <class Allocator = std::allocator<typename std::remove_const<T>::type>>
	array_proxy(std::vector<typename std::remove_const<T>::type, Allocator> const &data)
		: m_Size(static_cast<uint32_t>(data.size())), m_Data(data.data())
	{
	}

	array_proxy(std::initializer_list<T> const &data) : m_Size(static_cast<uint32_t>(data.end() - data.begin()))
	{
		m_Data = data.begin();
	}

	const T *data() const { return m_Data; }

	const T &at(size_t index) const { return m_Data[index]; }

	[[nodiscard]] bool has(size_t index) const { return m_Size > index; }

	const T *begin() const { return m_Data; }

	const T *end() const { return m_Data + m_Size; }

	const T &front() const
	{
		assert(m_Size > 0 && m_Data != nullptr);
		return *m_Data;
	}

	const T &back() const
	{
		assert(m_Size > 0 && m_Data != nullptr);
		return *(m_Data + m_Size - 1);
	}

	[[nodiscard]] bool empty() const { return (m_Size == 0); }

	[[nodiscard]] size_t size() const { return m_Size; }

	const T &operator[](const size_t idx) const { return m_Data[idx]; };

  private:
	size_t m_Size;
	const T *m_Data;
};
} // namespace rfw::utils

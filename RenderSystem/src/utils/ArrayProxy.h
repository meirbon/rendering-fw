#pragma once

#include <array>
#include <vector>
#include <cassert>

namespace rfw::utils
{
template <typename T> class ArrayProxy
{
  public:
	ArrayProxy(const std::vector<T> &data) noexcept : m_Size(data.size), m_Data(data.data()) {}
	template <size_t B> ArrayProxy(const std::array<T, B> &data) : m_Size(B), m_Data(data.data()) {}
	ArrayProxy(size_t count, const T *data) noexcept : m_Size(count), m_Data(data) {}

	ArrayProxy(const std::initializer_list<typename std::remove_reference<T>::type> &data) noexcept
		: m_Size(data.end() - data.begin()),
		  m_Data(data.begin())
	{
	}

	const T *data() const { return m_Data; }
	size_t size() const { return m_Size; }

	const T *begin() const noexcept { return m_Data; }

	const T *end() const noexcept { return m_Data + m_Size; }

	const T &front() const noexcept
	{
		assert(m_Size && m_Data);
		return *m_Data;
	}

	const T &back() const noexcept
	{
		assert(m_Size && m_Data);
		return *(m_Data + m_Size - 1);
	}

  private:
	size_t m_Size;
	const T *m_Data;
};
} // namespace rfw::utils
#pragma once

#include <deque>

namespace rfw::utils
{
template <typename T, size_t COUNT> class Averager
{
  public:
	Averager()
	{
		m_Index = 0;
		m_Minimum = 0;
		m_Maximum = 0;
		m_Samples.resize(COUNT, 0);
		m_Flipped = false;
	}

	void addSample(const T &value)
	{
		m_Dirty = true;
		if (m_Samples.size() > COUNT)
		{
			m_Flipped = true;
			m_Samples.pop_front();
		}
		m_Samples.push_back(value);
	}

	T getAverage()
	{
		if (m_Dirty)
		{

			m_Average = 0;
			const size_t s = m_Flipped ? std::min(COUNT, m_Samples.size()) : m_Index;
			for (size_t i = 0; i < s; i++)
				m_Average += m_Samples.at(i);
			m_Average /= T(COUNT);
			m_Dirty = false;
		}

		return m_Average;
	}

	T getAverage() const
	{
		T average = 0;
		if (m_Dirty)
		{

			const size_t s = m_Flipped ? std::min(COUNT, m_Samples.size()) : m_Index;
			for (size_t i = 0; i < s; i++)
				average += m_Samples.at(i);
			average /= T(COUNT);
		}

		return average;
	}

	T getMininum() const { return m_Minimum; }

	T getMaximum() const { return m_Maximum; }

	operator T() const { return getAverage(); }

	std::array<T, COUNT> data() const
	{
		std::array<T, COUNT> d{};
		for (size_t i = 0; i < COUNT; i++)
			d.at(i) = m_Samples.at(i);
		return d;
	}
	size_t size() const { return COUNT; }
	size_t byteSize() const { return COUNT * sizeof(T); }

  private:
	unsigned int m_Index;
	T m_Average, m_Minimum, m_Maximum;
	std::deque<T> m_Samples;
	bool m_Flipped, m_Dirty;
};
} // namespace rfw::utils
#pragma once

#include <random>
#include <glm/glm.hpp>

namespace rfw::utils
{
// http://www.rorydriscoll.com/2009/01/07/better-sampling/
class rng
{
  public:
	virtual ~rng() = default;

	virtual float rand(float range = 1.0f) { return rand_uint() * 2.3283064365387e-10f * range; }

	virtual unsigned int rand_uint() = 0;

	inline glm::vec3 sample_hemisphere(const float &r1, const float &r2)
	{
		const float r = sqrt(1.0f - r1 * r1);
		const float phi = 2.0f * 3.14159265358979323846f * r2;
		return glm::vec3(cos(phi) * r, r1, sin(phi) * r);
	}

	inline void create_coordinate_system(const glm::vec3 &N, glm::vec3 &Nt, glm::vec3 &Nb)
	{
		if (fabs(N.x) > fabs(N.y))
		{
			Nt = glm::vec3(N.z, 0.f, -N.x) / static_cast<float>(sqrt(N.x * N.x + N.z * N.z));
		}
		else
		{
			Nt = glm::vec3(0.f, -N.z, N.y) / static_cast<float>(sqrt(N.y * N.y + N.z * N.z));
		}

		Nb = glm::cross(N, Nt);
	}

	inline glm::vec3 local_to_world(const glm::vec3 &sample, const glm::vec3 &Nt, const glm::vec3 &Nb,
									const glm::vec3 &normal)
	{
		return glm::vec3(sample.x * Nb.x + sample.y * normal.x + sample.z * Nt.x,
						 sample.x * Nb.y + sample.y * normal.y + sample.z * Nt.y,
						 sample.x * Nb.z + sample.y * normal.z + sample.z * Nt.z);
	}

	inline glm::vec3 world_to_local_micro(const glm::vec3 &vec, const glm::vec3 &rDirection, glm::vec3 &u, glm::vec3 &v,
										  glm::vec3 &w)
	{
		w = vec;
		u = glm::normalize(glm::cross(fabs(vec.x) > fabs(vec.y) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0), w));
		v = glm::cross(w, u);
		const glm::vec3 wi = -rDirection;
		return glm::normalize(glm::vec3(glm::dot(u, wi), glm::dot(v, wi), glm::dot(w, wi)));
	}

	inline glm::vec3 local_to_world_micro(const glm::vec3 &wmLocal, const glm::vec3 &u, const glm::vec3 &v,
										  const glm::vec3 &w)
	{
		return u * wmLocal.x + v * wmLocal.y + w * wmLocal.z;
	}

	inline glm::vec3 point_on_hemisphere(const glm::vec3 &normal)
	{
		glm::vec3 Nt, Nb;
		create_coordinate_system(normal, Nt, Nb);
		const float r1 = rand(1.0f);
		const float r2 = rand(1.0f);
		const glm::vec3 sample = sample_hemisphere(r1, r2);
		return local_to_world(sample, Nt, Nb, normal);
	}
};
} // namespace rfw::utils
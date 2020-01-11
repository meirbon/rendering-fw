#pragma once

#import <Metal/Metal.h>

#include "MathIncludes.h"
#include "Structures.h"

namespace mtl
{
class Mesh
{
  public:
	Mesh();
	~Mesh();

	void set_geometry(const rfw::Mesh &mesh);
  private:
};
} // namespace mtl
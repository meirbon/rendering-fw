#include "shared.h"

kernel void finalize(write_only image2d_t outimg, // 0
					 global float4 *accumulator,  // 1
					 uint width,				  // 2
					 uint height,				  // 3
					 float scale				  // 4
)
{
	const int2 id = (int2)(get_global_id(0), get_global_id(1));
	const int2 dims = get_image_dim(outimg);

	if (id.x >= width || id.y >= height)
		return;

	const float4 color = accumulator[id.x + id.y * width] * scale;
	write_imagef(outimg, id, color);
}
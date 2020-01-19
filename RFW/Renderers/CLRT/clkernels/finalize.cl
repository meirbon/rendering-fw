#include "shared.h"

kernel void finalize(write_only image2d_t outimg, global float4 *accumulator, uint width, uint height, float scale)
{
	const int2 id = (int2)(get_global_id(0), get_global_id(1));
	const int2 dims = get_image_dim(outimg);

	if (id.x >= dims.x || id.y >= dims.y)
		return;

#if 1
	const float4 color = accumulator[id.x + id.y * width];
#else
	const float4 color = (float4)((float)id.x / dims.x, (float)id.y / dims.y, 0.2, 1);
#endif
	write_imagef(outimg, id, color);
}
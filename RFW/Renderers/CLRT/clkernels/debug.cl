
kernel void draw(write_only image2d_t outimg)
{
	const int2 id = make_int2(get_global_id(0), get_global_id(1));
	const int2 dims = get_image_dim(outimg);

	if (id.x >= dims.x || id.y >= dims.y)
		return;

	const float4 color = make_float4((float)id.x / dims.x, (float)id.y / dims.y, 0.2f, 1.0f);
	write_imagef(outimg, id, color);
}
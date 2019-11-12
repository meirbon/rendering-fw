#include <metal_stdlib>

using namespace metal;

vertex float4 v_simple(constant float4 *in [[buffer(0)]], uint vid [[vertex_id]]) { return in[vid]; }

fragment float4 f_simple(float4 in [[stage_in]]) { return float4(1, 0, 0, 1); }
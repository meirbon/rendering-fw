#pragma once

int intersect4(cpurt::RayPacket4 &packet, const glm::vec4 &p0, const glm::vec4 &p1, const glm::vec4 &p2, __m128 *store_mask, float epsilon = 1e-6f);

int intersect4(cpurt::RayPacket4 &packet, const glm::vec3 &p0, const glm::vec3 &edge1, const glm::vec3 &edge2, __m128 *store_mask,
					  float epsilon = 1e-6f);
int intersect4(cpurt::RayPacket4 &packet, const glm::vec3 &p0, const glm::vec3 &edge1, const glm::vec3 &edge2, glm::vec2 *bary4, __m128 *store_mask,
					  float epsilon = 1e-6f);

int intersect4(cpurt::RayPacket4 &packet, const rfw::bvh::BVHNode &node);
// static int intersect4(cpurt::RayPacket4 &packet, const rfw::bvh::MBVHNode &node, __m128 *hit_mask);

// static int traverse_mbvh(cpurt::RayPacket4 &packet, float t_min, const rfw::bvh::MBVHNode *nodes, const unsigned int *primIndices, const glm::vec3 *p0s,
//						 const glm::vec3 *edge1s, const glm::vec3 *edge2s, __m128 *hit_mask);

int traverse_bvh(cpurt::RayPacket4 &packet, float t_min, const rfw::bvh::BVHNode *nodes, const unsigned int *primIndices,
						const rfw::bvh::rfwMesh **meshes, __m128 *hit_mask);

int traverse_bvh(cpurt::RayPacket4 &packet, float t_min, const rfw::bvh::BVHNode *nodes, const unsigned int *primIndices, const glm::vec3 *p0s,
						const glm::vec3 *edge1s, const glm::vec3 *edge2s, __m128 *hit_mask);
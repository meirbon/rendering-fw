#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <tiny_gltf.h>

#include "gLTFObject.h"

#include "../utils/File.h"

rfw::SceneAnimation creategLTFAnim(rfw::SceneObject *object, tinygltf::Animation &gltfAnim, tinygltf::Model &gltfModel, const int nodeBase);

rfw::MeshSkin convertSkin(const tinygltf::Skin skin, const tinygltf::Model model)
{
	rfw::MeshSkin s = {};
	s.name = skin.name;
	s.joints.reserve(skin.joints.size());
	for (auto joint : skin.joints)
		s.joints.emplace_back(joint);

	if (skin.inverseBindMatrices > -1)
	{
		const auto &accessor = model.accessors.at(skin.inverseBindMatrices);
		const auto &bufferView = model.bufferViews.at(accessor.bufferView);
		const auto &buffer = model.buffers.at(bufferView.buffer);

		s.inverseBindMatrices.resize(accessor.count);
		memcpy(s.inverseBindMatrices.data(), &buffer.data.at(accessor.byteOffset + bufferView.byteOffset), accessor.count * sizeof(glm::mat4));
		//for (auto& ibm : s.inverseBindMatrices)
		//	ibm = rowMajor4(ibm);

		s.jointMatrices.resize(accessor.count, glm::mat4(1.0f));
	}

	return s;
}

rfw::SceneNode createNode(rfw::gLTFObject &object, const tinygltf::Node &node)
{
	auto n = rfw::SceneNode(&object.scene, node.name, {});
	if (node.mesh > -1)
	{
		n.meshIDs.push_back(node.mesh);
		if (node.skin > -1)
			n.skinIDs.push_back(node.skin);
		else
			n.skinIDs.push_back(-1);
	}

	if (!n.meshIDs.empty())
	{
		const auto morphTargets = glm::max(object.scene.meshes.at(n.meshIDs.at(0)).poses.size(), size_t(1)) - 1;
		if (morphTargets > 0)
		{
			n.weights.resize(morphTargets, 0.0f);
		}
	}

	if (!node.children.empty())
	{
		n.childIndices.reserve(node.children.size());
		for (size_t s = node.children.size(), i = 0; i < s; i++)
			n.childIndices.push_back(node.children.at(i));
	}

	if (node.matrix.size() == 16)
	{
		for (size_t i = 0; i < 4; i++)
			n.matrix[i] = vec4(glm::make_vec4(&node.matrix.at(i * 4u)));
	}

	if (node.translation.size() == 3)
	{
		// the GLTF node contains a translation
		n.translation = vec3(node.translation[0], node.translation[1], node.translation[2]);
	}

	if (node.rotation.size() == 4)
	{
		// the GLTF node contains a rotation
		n.rotation = quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
	}

	if (node.scale.size() == 3)
	{
		// the GLTF node contains a scale
		n.scale = vec3(node.scale[0], node.scale[1], node.scale[2]);
	}

	n.calculateTransform();
	return n;
}

rfw::gLTFObject::gLTFObject(std::string_view filename, MaterialList *matList, uint ID, const glm::mat4 &matrix, int material) : file(filename.data())
{
	using namespace tinygltf;

	Model model;
	TinyGLTF loader;
	std::string err, warn;

	bool ret = false;
	if (utils::string::ends_with(filename, ".glb"))
		ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename.data()); // for binary glTF(.glb)
	else
		ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename.data());

	if (!warn.empty())
		WARNING("%s", warn.data());
	if (!err.empty())
	{
		WARNING("%s", err.data());
		throw LoadException(err);
	}

	if (!ret)
	{
		const std::string message = std::string("Could not load \"") + filename.data() + "\"";
		WARNING("%s", message.data());
		throw LoadException(message);
	}

	m_BaseMaterialIdx = matList->getMaterials().size();
	const auto baseTextureIdx = matList->getTextures().size();

	for (size_t i = 0; i < model.materials.size(); i++)
	{
		const auto &tinyMat = model.materials.at(i);

		HostMaterial mat = {};
		mat.name = tinyMat.name;
		for (const auto &value : tinyMat.values)
		{
			{
				if (value.first == "baseColorFactor")
				{
					tinygltf::Parameter p = value.second;
					mat.color = vec3(p.number_array[0], p.number_array[1], p.number_array[2]);
				}
			}
			if (value.first == "metallicFactor")
			{
				if (value.second.has_number_value)
				{
					mat.metallic = (float)value.second.number_value;
				}
			}
			if (value.first == "roughnessFactor")
			{
				if (value.second.has_number_value)
				{
					mat.roughness = (float)value.second.number_value;
				}
			}
			if (value.first == "baseColorTexture")
			{
				for (auto &item : value.second.json_double_value)
				{
					if (item.first == "index")
						mat.map[TEXTURE0].textureID = (int)item.second + baseTextureIdx;
				}
			}
		}

		matList->add(mat);
	}

	for (size_t i = 0; i < model.textures.size(); i++)
	{
		const auto &gltfTex = model.textures.at(i);
		const auto &gltfImage = model.images.at(gltfTex.source);

		auto texture = Texture((uint *)gltfImage.image.data(), gltfImage.width, gltfImage.height);
		matList->add(texture);
	}

	scene.skins.resize(model.skins.size());
	for (size_t i = 0; i < model.skins.size(); i++)
		scene.skins.at(i) = convertSkin(model.skins.at(i), model);

	scene.animations.resize(model.animations.size());
	for (size_t i = 0; i < model.animations.size(); i++)
		scene.animations.at(i) = creategLTFAnim(&scene, model.animations.at(i), model, 0);

	scene.meshes.resize(model.meshes.size());

	for (size_t i = 0; i < model.meshes.size(); i++)
	{
		const auto &mesh = model.meshes.at(i);

		// Add mesh information to object
		rfw::SceneMesh &m = scene.meshes.at(i);
		m.object = &scene;
		m.vertexOffset = static_cast<uint>(scene.baseVertices.size());

		for (size_t s = mesh.primitives.size(), j = 0; j < s; j++)
		{
			const Primitive &prim = mesh.primitives.at(j);

			const Accessor &accessor = model.accessors.at(prim.indices);
			const BufferView &view = model.bufferViews.at(accessor.bufferView);
			const Buffer &buffer = model.buffers.at(view.buffer);
			const unsigned char *a = buffer.data.data() + view.byteOffset + accessor.byteOffset;
			const int byteStride = accessor.ByteStride(view);
			const size_t count = accessor.count;

			std::vector<int> tmpIndices;
			std::vector<glm::vec3> tmpNormals, tmpVertices;
			std::vector<glm::vec2> tmpUvs;
			std::vector<glm::uvec4> tmpJoints;
			std::vector<glm::vec4> tmpWeights;

			switch (accessor.componentType)
			{
			case (TINYGLTF_COMPONENT_TYPE_BYTE):
				for (int k = 0; k < count; k++, a += byteStride)
					tmpIndices.push_back(*((char *)a));
				break;
			case (TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE):
				for (int k = 0; k < count; k++, a += byteStride)
					tmpIndices.push_back(*((unsigned char *)a));
				break;
			case (TINYGLTF_COMPONENT_TYPE_SHORT):
				for (int k = 0; k < count; k++, a += byteStride)
					tmpIndices.push_back(*((short *)a));
				break;
			case (TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT):
				for (int k = 0; k < count; k++, a += byteStride)
					tmpIndices.push_back(*((unsigned short *)a));
				break;
			case (TINYGLTF_COMPONENT_TYPE_INT):
				for (int k = 0; k < count; k++, a += byteStride)
					tmpIndices.push_back(*((int *)a));
				break;
			case (TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT):
				for (int k = 0; k < count; k++, a += byteStride)
					tmpIndices.push_back(*((unsigned int *)a));
				break;
			default:
				break;
			}

			if (prim.mode == TINYGLTF_MODE_TRIANGLE_FAN)
			{
				auto fan = move(tmpIndices);
				tmpIndices.clear();
				for (size_t sj = fan.size(), p = 2; p < sj; p++)
				{
					tmpIndices.push_back(fan.at(0));
					tmpIndices.push_back(fan.at(p - 1));
					tmpIndices.push_back(fan.at(p));
				}
			}
			else if (prim.mode == TINYGLTF_MODE_TRIANGLE_STRIP)
			{
				auto strip = move(tmpIndices);
				tmpIndices.clear();
				for (size_t sj = strip.size(), p = 2; p < sj; p++)
				{
					tmpIndices.push_back(strip.at(p - 2));
					tmpIndices.push_back(strip.at(p - 1));
					tmpIndices.push_back(strip.at(p));
				}
			}
			else if (prim.mode != TINYGLTF_MODE_TRIANGLES)
				continue;

			for (const auto &attribute : prim.attributes)
			{
				const Accessor &attribAccessor = model.accessors.at(attribute.second);
				const BufferView &bufferView = model.bufferViews.at(attribAccessor.bufferView);
				const Buffer &buffer = model.buffers.at(bufferView.buffer);
				const unsigned char *a = buffer.data.data() + bufferView.byteOffset + attribAccessor.byteOffset;
				const int byteStride = attribAccessor.ByteStride(bufferView);
				const size_t count = attribAccessor.count;

				if (attribute.first == "POSITION")
				{
					if (attribAccessor.type == TINYGLTF_TYPE_VEC3)
					{
						if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
						{
							for (size_t p = 0; p < count; p++, a += byteStride)
							{
								tmpVertices.push_back(*((glm::vec3 *)a));
							}
						}
						else if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE)
						{
							// WARNING("%s", "Double precision positions are not supported (yet).");
							for (size_t p = 0; p < count; p++, a += byteStride)
							{
								tmpVertices.emplace_back(*((glm::dvec3 *)a));
							}
						}
					}
					else
					{
						throw LoadException("Unsupported position definition in gLTF file.");
					}
				}
				else if (attribute.first == "NORMAL")
				{
					if (attribAccessor.type == TINYGLTF_TYPE_VEC3)
					{
						if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
						{
							for (size_t f = 0; f < count; f++, a += byteStride)
							{
								tmpNormals.push_back(*((glm::vec3 *)a));
							}
						}
						else if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE)
						{
							// WARNING("%s", "Double precision positions are not supported (yet).");
							for (size_t f = 0; f < count; f++, a += byteStride)
							{
								tmpNormals.emplace_back(*((glm::dvec3 *)a));
							}
						}
					}
					else
					{
						throw LoadException("Unsupported normal definition in gLTF file.");
					}
				}
				else if (attribute.first == "TANGENT")
					continue;
				else if (attribute.first == "TEXCOORD_0")
				{
					if (attribAccessor.type == TINYGLTF_TYPE_VEC2)
					{
						if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
						{
							for (size_t f = 0; f < count; f++, a += byteStride)
							{
								tmpUvs.push_back(*((glm::vec2 *)a));
							}
						}
						else if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE)
						{
							// WARNING("%s", "Double precision normals are not supported (yet).");
							for (size_t f = 0; f < count; f++, a += byteStride)
							{
								tmpUvs.emplace_back(*((glm::dvec2 *)a));
							}
						}
					}
					else
					{
						throw LoadException("Unsupported UV definition in gLTF file.");
					}
				}
				else if (attribute.first == "TEXCOORD_1")
					continue;
				else if (attribute.first == "COLOR_0")
					continue;
				else if (attribute.first == "JOINTS_0")
				{
					if (attribAccessor.type == TINYGLTF_TYPE_VEC4)
					{
						if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
						{
							using unshort = unsigned short;

							for (size_t f = 0; f < count; f++, a += byteStride)
							{
								tmpJoints.emplace_back(*((unshort *)a), *((unshort *)(a + 2)), *((unshort *)(a + 4)), *((unshort *)(a + 6)));
							}
						}
						else if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
						{
							using uchar = unsigned char;
							for (size_t f = 0; f < count; f++, a += byteStride)
							{
								tmpJoints.emplace_back(*((uchar *)a), *((uchar *)(a + 1)), *((uchar *)(a + 2)), *((uchar *)(a + 3)));
							}
						}
						else
						{
							throw LoadException("Expected unsigned shorts or bytes for joints in gLTF file.");
						}
					}
					else
					{
						throw LoadException("Unsupported joint definition in gLTF file.");
					}
				}
				else if (attribute.first == "WEIGHTS_0")
				{
					if (attribAccessor.type == TINYGLTF_TYPE_VEC4)
					{
						if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
						{
							for (size_t f = 0; f < count; f++, a += byteStride)
							{
								glm::vec4 w4;
								memcpy(&w4, a, sizeof(glm::vec4));
								float norm = 1.0f / (w4.x + w4.y + w4.z + w4.w);
								w4 *= norm;
								tmpWeights.push_back(w4);
							}
						}
						else if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE)
						{
							// WARNING("%s", "Double precision weights are not supported (yet).");
							for (size_t f = 0; f < count; f++, a += byteStride)
							{
								glm::dvec4 w4;
								memcpy(&w4, a, sizeof(glm::dvec4));
								const float norm = 1.0 / (w4.x + w4.y + w4.z + w4.w);
								w4 *= norm;
								tmpWeights.emplace_back(w4);
							}
						}
					}
					else
					{
						throw LoadException("Unsupported weight definition in gLTF file.");
					}
				}
				else
				{
					WARNING("Unknown property: \"%s\"", attribute.first.data());
				}
			}

			std::vector<rfw::SceneMesh::Pose> tmpPoses;
			tmpPoses.reserve(mesh.weights.size() + 1);

			// Store base pose if morph target present
			if (!mesh.weights.empty())
			{
				tmpPoses.emplace_back();
				for (size_t sv = tmpVertices.size(), n = 0; n < sv; n++)
				{
					tmpPoses.at(0).positions.push_back(tmpVertices.at(n));
					tmpPoses.at(0).normals.push_back(tmpNormals.at(n));
				}
			}

			for (size_t o = 0; o < mesh.weights.size(); o++)
			{
				tmpPoses.emplace_back();
				auto &pose = tmpPoses.at(tmpPoses.size() - 1);

				for (const auto &target : prim.targets.at(o))
				{
					const Accessor &acc = model.accessors.at(target.second);
					const BufferView &vi = model.bufferViews.at(acc.bufferView);
					const auto *va = (const float *)(model.buffers.at(vi.buffer).data.data() + vi.byteOffset + acc.byteOffset);

					for (size_t p = 0; p < acc.count; p++)
					{
						const auto v = glm::vec3(va[p * 3], va[p * 3 + 1], va[p * 3 + 2]);

						if (target.first == "POSITION")
							pose.positions.push_back(v);
						else if (target.first == "NORMAL")
							pose.normals.push_back(v);
					}
				}
			}

			addPrimitive(m, tmpIndices, tmpVertices, tmpNormals, tmpUvs, tmpPoses, tmpJoints, tmpWeights,
						 prim.material >= 0 ? static_cast<int>(prim.material + m_BaseMaterialIdx) : 0);
		}

		m.vertexCount = scene.baseVertices.size() - m.vertexOffset;
		m.faceCount = m.vertexCount / 3;
		m.faceOffset = m.vertexOffset / 3;
	}

	const bool hasTransform = matrix != glm::mat4(1.0f);

	if (model.scenes.size() > 1)
		WARNING("gLTF files with more than 1 scene are not supported (yet).");

	scene.nodes.reserve(model.nodes.size() + 1);
	tinygltf::Scene &gltfScene = model.scenes.at(0);

	for (size_t s = model.nodes.size(), i = 0; i < s; i++)
	{
		const auto &node = model.nodes.at(i);
		scene.nodes.emplace_back(createNode(*this, node));
	}

	for (size_t i = 0; i < gltfScene.nodes.size(); i++)
		scene.rootNodes.push_back(i);

	if (hasTransform)
	{
		for (int i : scene.rootNodes)
		{
			auto &node = scene.nodes.at(i);
			node.matrix = matrix * node.matrix;
			node.transformed = true;
		}
	}

	scene.vertices.resize(scene.baseVertices.size(), vec4(0, 0, 0, 1));
	scene.normals.resize(scene.baseNormals.size(), vec3(0.0f));

	scene.transformTo(0.0f);

	// Update triangle data that only has to be calculated once
	scene.updateTriangles(matList, texCoords);

	utils::logger::log("Loaded file: %s with %u vertices and %u triangles", filename.data(), scene.vertices.size(), scene.triangles.size());
}

void rfw::gLTFObject::transformTo(float timeInSeconds) { scene.transformTo(timeInSeconds); }

rfw::Triangle *rfw::gLTFObject::getTriangles() { return scene.triangles.data(); }

glm::vec4 *rfw::gLTFObject::getVertices() { return scene.vertices.data(); }

rfw::Mesh rfw::gLTFObject::getMesh() const
{
	auto mesh = rfw::Mesh();
	mesh.vertices = scene.vertices.data();
	mesh.normals = scene.normals.data();
	mesh.triangles = scene.triangles.data();
	mesh.indices = nullptr;
	mesh.vertexCount = scene.vertices.size();
	mesh.triangleCount = scene.vertices.size() / 3;
	mesh.texCoords = texCoords.data();
	return mesh;
}

bool rfw::gLTFObject::isAnimated() const { return !scene.animations.empty(); }

uint rfw::gLTFObject::getAnimationCount() const { return uint(scene.animations.size()); }

void rfw::gLTFObject::setAnimation(uint index)
{
	// TODO
}

uint rfw::gLTFObject::getMaterialForPrim(uint primitiveIdx) const { return scene.materialIndices.at(primitiveIdx); }

std::vector<uint> rfw::gLTFObject::getLightIndices(const std::vector<bool> &matLightFlags) const
{
	std::vector<uint> indices;
	for (const auto &mesh : scene.meshes)
	{
		const size_t offset = indices.size();
		if (matLightFlags.at(mesh.matID))
		{
			const auto s = mesh.vertexCount / 3;
			const auto o = mesh.vertexOffset / 3;
			indices.resize(offset + s);
			for (uint i = 0; i < s; i++)
				indices.at(offset + i) = o + i;
		}
	}

	return indices;
}

void rfw::gLTFObject::addPrimitive(rfw::SceneMesh &mesh, const std::vector<int> &indices, const std::vector<glm::vec3> &vertices,
								   const std::vector<glm::vec3> &normals, const std::vector<glm::vec2> &uvs, const std::vector<rfw::SceneMesh::Pose> &poses,
								   const std::vector<glm::uvec4> &joints, const std::vector<glm::vec4> &weights, const int materialIdx)
{
	using namespace glm;

	// Allocate data
	const auto triangleOffset = scene.triangles.size();

	scene.materialIndices.resize(scene.materialIndices.size() + (indices.size() / 3));
	scene.triangles.resize(scene.triangles.size() + (indices.size() / 3));

	scene.baseVertices.reserve(scene.baseVertices.size() + indices.size());
	scene.baseNormals.reserve(scene.baseNormals.size() + indices.size());
	texCoords.reserve(texCoords.size() + indices.size());

	mesh.poses.resize(poses.size());
	for (size_t i = 0; i < mesh.poses.size(); i++)
	{
		const auto &origPose = poses.at(i);
		auto &pose = mesh.poses.at(i);

		pose.positions.reserve(indices.size());
		pose.normals.reserve(indices.size());

		for (int idx : indices)
		{
			pose.positions.push_back(origPose.positions.at(idx));
			pose.normals.emplace_back(origPose.normals.at(idx));
		}
	}

	if (!joints.empty())
	{
		mesh.joints.reserve(indices.size());
		mesh.weights.reserve(indices.size());

		for (size_t s = indices.size(), i = 0; i < s; i++)
		{
			const auto idx = indices.at(i);

			mesh.joints.push_back(joints.at(idx));
			mesh.weights.push_back(weights.at(idx));
		}
	}

	// Add per-vertex data
	for (size_t s = indices.size(), i = 0; i < s; i++)
	{
		const auto idx = indices.at(i);

		scene.baseVertices.emplace_back(vertices.at(idx), 1.0f);
		scene.baseNormals.push_back(normals.at(idx));
		if (!uvs.empty())
			texCoords.push_back(uvs.at(idx));
		else
			texCoords.emplace_back(0.0f);
	}

	// Add per-face data
	for (size_t s = indices.size() / 3, triIdx = triangleOffset, i = 0; i < s; i++, triIdx++)
	{
		const auto idx = uvec3(indices.at(i * 3 + 0), indices.at(i * 3 + 1), indices.at(i * 3 + 2));
		auto &tri = scene.triangles.at(triIdx);
		scene.materialIndices.at(triIdx) = materialIdx;

		const auto v0 = vertices.at(idx.x);
		const auto v1 = vertices.at(idx.y);
		const auto v2 = vertices.at(idx.z);

		const auto &n0 = normals.at(idx.x);
		const auto &n1 = normals.at(idx.y);
		const auto &n2 = normals.at(idx.z);

		const vec3 N = normalize(cross(v1 - v0, v2 - v0));
		tri.Nx = N.x;
		tri.Ny = N.y;
		tri.Nz = N.z;

		tri.vertex0 = v0;
		tri.vertex1 = v1;
		tri.vertex2 = v2;

		tri.vN0 = n0;
		tri.vN1 = n1;
		tri.vN2 = n2;

		if (!uvs.empty())
		{
			tri.u0 = uvs.at(idx.x).x;
			tri.u1 = uvs.at(idx.y).x;
			tri.u2 = uvs.at(idx.z).x;

			tri.v0 = uvs.at(idx.x).y;
			tri.v1 = uvs.at(idx.y).y;
			tri.v2 = uvs.at(idx.z).y;
		}

		tri.material = materialIdx;
	}
}

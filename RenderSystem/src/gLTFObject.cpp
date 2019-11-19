#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <tiny_gltf.h>

#include "gLTFObject.h"

#include "utils/File.h"

#include "gLTFAnimation.h"

rfw::gLTFAnimation creategLTFAnim(tinygltf::Animation &gltfAnim, tinygltf::Model &gltfModel, const int nodeBase);

rfw::gTLFObject::gLTFSkin convertSkin(const tinygltf::Skin skin, const tinygltf::Model model, const int nodeBase)
{
	rfw::gTLFObject::gLTFSkin s = {};

	s.name = skin.name;
	if (skin.skeleton == -1)
		s.skeletonRoot = 0;
	else
		s.skeletonRoot = s.skeletonRoot;

	s.skeletonRoot = s.skeletonRoot + nodeBase;

	s.joints = skin.joints;
	for (int &j : s.joints)
		j = j + nodeBase;

	if (skin.inverseBindMatrices > -1)
	{
		const auto &accessor = model.accessors.at(skin.inverseBindMatrices);
		const auto &bufferView = model.bufferViews.at(accessor.bufferView);
		const auto &buffer = model.buffers.at(bufferView.buffer);

		s.inverseBindMatrices.resize(accessor.count);
		memcpy(s.inverseBindMatrices.data(), &buffer.data.at(accessor.byteOffset + bufferView.byteOffset),
			   accessor.count * sizeof(glm::mat4));
		s.jointMatrices.resize(accessor.count);
	}

	return s;
}

rfw::gTLFObject::gLTFNode createNode(const rfw::gTLFObject &object, const tinygltf::Node &node, const int nodebase,
									 const int meshBase, const int skinBase)
{
	rfw::gTLFObject::gLTFNode n = {};

	n.name = node.name;
	n.meshID = node.mesh == -1 ? -1 : node.mesh + meshBase;
	n.skinID = node.skin == -1 ? -1 : node.skin + skinBase;
	if (n.meshID != -1)
	{
		const auto morphTargets = object.m_Meshes.at(n.meshID).poseCount;
		if (morphTargets > 0)
			n.weights.resize(morphTargets, 0.0f);
	}

	for (size_t s = node.children.size(), i = 0; i < s; i++)
	{
		n.childIndices.push_back(node.children.at(i) + nodebase);
	}

	bool buildFromTRS = false;
	if (node.matrix.size() == 16)
	{
		memcpy(value_ptr(n.matrix), node.matrix.data(), 16 * sizeof(float));
	}
	if (node.translation.size() == 3)
	{
		// the GLTF node contains a translation
		n.translation = vec3(node.translation[0], node.translation[1], node.translation[2]);
		buildFromTRS = true;
	}
	if (node.rotation.size() == 4)
	{
		// the GLTF node contains a rotation
		n.rotation = quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
		buildFromTRS = true;
	}
	if (node.scale.size() == 3)
	{
		// the GLTF node contains a scale
		n.scale = vec3(node.scale[0], node.scale[1], node.scale[2]);
		buildFromTRS = true;
	}

	if (buildFromTRS)
		n.updateTransform();

	n.prepareLights();

	return n;
}

rfw::gTLFObject::gTLFObject(std::string_view filename, MaterialList *matList, uint ID, const glm::mat4 &matrix,
							bool normalize, int material)
	: file(filename.data())
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
			if (value.first == "baseColorFactor")
			{
				tinygltf::Parameter p = value.second;
				mat.color = vec3(p.number_array[0], p.number_array[1], p.number_array[2]);
			}
			if (value.first == "metallicFactor")
				if (value.second.has_number_value)
				{
					mat.metallic = (float)value.second.number_value;
				}
			if (value.first == "roughnessFactor")
				if (value.second.has_number_value)
				{
					mat.roughness = (float)value.second.number_value;
				}
			if (value.first == "baseColorTexture")
				for (auto &item : value.second.json_double_value)
				{
					if (item.first == "index")
						mat.map[TEXTURE0].textureID = (int)item.second + baseTextureIdx;
				}
			// TODO: do a better automatic conversion.
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

	m_Skins.resize(model.skins.size());
	for (size_t i = 0; i < model.skins.size(); i++)
		m_Skins.at(i) = convertSkin(model.skins.at(i), model, 0);

	m_Animations.resize(model.animations.size());
	for (size_t i = 0; i < model.animations.size(); i++)
		m_Animations.at(i) = creategLTFAnim(model.animations.at(i), model, 0);

	m_Meshes.resize(model.meshes.size());
	for (size_t i = 0; i < model.meshes.size(); i++)
	{
		const auto &m = m_Meshes.at(i);
		const auto &mesh = model.meshes.at(i);

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
				for (size_t s = fan.size(), i = 2; i < s; i++)
				{
					tmpIndices.push_back(fan.at(0));
					tmpIndices.push_back(fan.at(i - 1));
					tmpIndices.push_back(fan.at(i));
				}
			}
			else if (prim.mode == TINYGLTF_MODE_TRIANGLE_STRIP)
			{
				auto strip = move(tmpIndices);
				tmpIndices.clear();
				for (size_t s = strip.size(), i = 2; i < s; i++)
				{
					tmpIndices.push_back(strip.at(i - 2));
					tmpIndices.push_back(strip.at(i - 1));
					tmpIndices.push_back(strip.at(i));
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
					// const auto boundsMin = glm::vec3(attribAccessor.minValues[0], attribAccessor.minValues[1],
					//								 attribAccessor.minValues[2]);
					// const auto boundsMax = glm::vec3(attribAccessor.maxValues[0], attribAccessor.maxValues[1],
					//								 attribAccessor.maxValues[2]);

					if (attribAccessor.type == TINYGLTF_TYPE_VEC3)
					{
						if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
						{
							for (size_t i = 0; i < count; i++, a += byteStride)
							{
								tmpVertices.push_back(*((glm::vec3 *)a));
							}
						}
						else if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE)
						{
							WARNING("%s", "Double precision positions are not supported (yet).");
							for (size_t i = 0; i < count; i++, a += byteStride)
							{
								tmpVertices.push_back(glm::vec3(*((glm::dvec3 *)a)));
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
							for (size_t i = 0; i < count; i++, a += byteStride)
							{
								tmpNormals.push_back(*((glm::vec3 *)a));
							}
						}
						else if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE)
						{
							WARNING("%s", "Double precision positions are not supported (yet).");
							for (size_t i = 0; i < count; i++, a += byteStride)
							{
								tmpNormals.push_back(glm::vec3(*((glm::dvec3 *)a)));
							}
						}
					}
					else
					{
						throw LoadException("Unsupported normal definition in gLTF file.");
					}
				}
				else if (attribute.first == "TANGENT")
				{
					WARNING("Tangents are not yet supported in gLTF file.");
					continue;
				}
				else if (attribute.first == "TEXCOORD_0")
				{
					if (attribAccessor.type == TINYGLTF_TYPE_VEC2)
					{
						if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
						{
							for (size_t i = 0; i < count; i++, a += byteStride)
							{
								tmpUvs.push_back(*((glm::vec2 *)a));
							}
						}
						else if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE)
						{
							WARNING("%s", "Double precision normals are not supported (yet).");
							for (size_t i = 0; i < count; i++, a += byteStride)
							{
								tmpUvs.push_back(glm::vec2(*((glm::dvec2 *)a)));
							}
						}
					}
					else
					{
						throw LoadException("Unsupported UV definition in gLTF file.");
					}
				}
				else if (attribute.first == "TEXCOORD_1")
				{
					// TODO
					continue;
				}
				else if (attribute.first == "COLOR_0")
				{
					// TODO
					continue;
				}
				else if (attribute.first == "JOINTS_0")
				{
					if (attribAccessor.type == TINYGLTF_TYPE_VEC4)
					{
						if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
						{
							using ushort = unsigned short;

							for (size_t i = 0; i < count; i++, a += byteStride)
							{
								tmpJoints.push_back(glm::uvec4(*((ushort *)a), *((ushort *)(a + 2)),
															   *((ushort *)(a + 4)), *((ushort *)(a + 6))));
							}
						}
						else if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
						{
							using uchar = unsigned char;
							for (size_t i = 0; i < count; i++, a += byteStride)
							{
								tmpJoints.push_back(glm::uvec4(*((uchar *)a), *((uchar *)(a + 1)), *((uchar *)(a + 2)),
															   *((uchar *)(a + 3))));
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
							for (size_t i = 0; i < count; i++, a += byteStride)
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
							WARNING("%s", "Double precision weights are not supported (yet).");
							for (size_t i = 0; i < count; i++, a += byteStride)
							{
								glm::vec4 w4;
								memcpy(&w4, a, sizeof(glm::vec4));
								float norm = 1.0f / (w4.x + w4.y + w4.z + w4.w);
								w4 *= norm;
								tmpWeights.push_back(w4);
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
					WARNING("Unknown property: \"%s\"", attribute.first);
				}
			}

			std::vector<gLTFPose> tmpPoses;
			if (!mesh.weights.empty())
			{
				tmpPoses.emplace_back();
				for (size_t s = tmpVertices.size(), i = 0; i < s; i++)
				{
					tmpPoses.at(0).positions.push_back(tmpVertices.at(i));
					tmpPoses.at(0).normals.push_back(tmpNormals.at(i));
					// TODO
					tmpPoses.at(0).tangents.push_back(glm::vec3(0));
				}
			}

			for (size_t j = 0; j < mesh.weights.size(); j++)
			{
				tmpPoses.emplace_back();
				for (const auto &target : prim.targets.at(j))
				{
					const Accessor &accessor = model.accessors.at(target.second);
					const BufferView &view = model.bufferViews.at(accessor.bufferView);
					const float *a = (const float *)(model.buffers.at(view.buffer).data.data() + view.byteOffset +
													 accessor.byteOffset);

					for (size_t m = 0; m < accessor.count; m++)
					{
						const auto v = glm::vec3(a[m * 3], a[m * 3 + 1], a[m * 3 + 2]);
						if (target.first == "POSITION")
							tmpPoses.at(i + 1).positions.push_back(v);
						else if (target.first == "NORMAL")
							tmpPoses.at(i + 1).normals.push_back(v);
						else if (target.first == "TANGENT")
							tmpPoses.at(i + 1).tangents.push_back(v);
					}
				}
			}

			build(tmpIndices, tmpVertices, tmpNormals, tmpUvs, tmpPoses, tmpJoints, tmpWeights,
				  prim.material > 0 ? prim.material : 0);
		}
	}

	const bool hasTransform = matrix != glm::mat4(1.0f);

	if (model.scenes.size() > 1)
		WARNING("gLTF files with more than 1 scene are not supported (yet).");
	Scene &gltfScene = model.scenes.at(0);
	if (hasTransform)
	{
		gLTFNode node = {};
		node.localTransform = matrix;
		node.ID = 0;
		m_Nodes.emplace_back(node);
	}

	m_Nodes.resize(m_Nodes.size() + model.nodes.size());
	for (size_t s = model.nodes.size(), i = 0; i < s; i++)
	{
		const Node &gltfNode = model.nodes.at(i);
		gLTFNode node = {};
		m_Nodes.at(i) = createNode(*this, gltfNode, 0, 0, 0);
	}

	m_CurrentVertices = m_BaseVertices;
	m_CurrentNormals = m_BaseNormals;
}

void rfw::gTLFObject::transformTo(float timeInSeconds) {}

rfw::Triangle *rfw::gTLFObject::getTriangles() { return m_Triangles.data(); }

glm::vec4 *rfw::gTLFObject::getVertices() { return m_CurrentVertices.data(); }

rfw::Mesh rfw::gTLFObject::getMesh() const
{
	auto mesh = rfw::Mesh();
	mesh.vertices = m_CurrentVertices.data();
	mesh.normals = m_CurrentNormals.data();
	mesh.triangles = m_Triangles.data();
	mesh.indices = nullptr;
	mesh.vertexCount = m_CurrentVertices.size();
	mesh.triangleCount = m_CurrentVertices.size() / 3;
	return mesh;
}

bool rfw::gTLFObject::isAnimated() const { return !m_Animations.empty(); }

uint rfw::gTLFObject::getAnimationCount() const { return uint(m_Animations.size()); }

void rfw::gTLFObject::setAnimation(uint index)
{
	// TODO
}

uint rfw::gTLFObject::getMaterialForPrim(uint primitiveIdx) const
{
	// TODO
	return 0;
}

std::vector<uint> rfw::gTLFObject::getLightIndices(const std::vector<bool> &matLightFlags) const
{
	// TODO
	return std::vector<uint>();
}

void rfw::gTLFObject::gLTFMesh::setPose(rfw::gTLFObject &object, const gLTFSkin &skin)
{
	using namespace glm;

	vec4 *baseVertex = &object.m_CurrentVertices.at(vertexOffset);
	vec3 *baseNormal = &object.m_CurrentNormals.at(vertexOffset);
	memcpy(baseVertex, &object.m_BaseVertices.at(vertexOffset), vertexCount * sizeof(glm::vec4));
	memcpy(baseNormal, &object.m_BaseNormals.at(vertexOffset), vertexCount * sizeof(glm::vec3));

	for (uint i = 0; i < vertexCount; i++)
	{
		const auto &j4 = object.m_Joints.at(jointOffset + i);
		const auto &w4 = object.m_Weights.at(weightOffset + i);

		mat4 skinMatrix = w4.x * skin.jointMatrices.at(j4.x);
		skinMatrix += w4.y * skin.jointMatrices[j4.y];
		skinMatrix += w4.z * skin.jointMatrices[j4.z];
		skinMatrix += w4.w * skin.jointMatrices[j4.w];

		baseVertex[i] = skinMatrix * baseVertex[i];
		baseNormal[i] = normalize(mat3(skinMatrix) * baseVertex[i]);
	}

	const auto offset = vertexOffset / 3;
	for (uint s = (vertexCount / 3), i = 0; i < s; i++)
	{
		auto &tri = object.m_Triangles.at(i + offset);
		tri.vertex0 = baseVertex[i * 3 + 0];
		tri.vertex1 = baseVertex[i * 3 + 1];
		tri.vertex2 = baseVertex[i * 3 + 2];

		tri.vN0 = normalize(baseNormal[i * 3 + 0]);
		tri.vN1 = normalize(baseNormal[i * 3 + 1]);
		tri.vN2 = normalize(baseNormal[i * 3 + 2]);
	}

	object.m_HasUpdated = true;
}

void rfw::gTLFObject::gLTFMesh::setPose(rfw::gTLFObject &object, const std::vector<float> &weights)
{
	assert(weights.size() == object.m_Poses.size() - 1);
	const auto weightCount = weights.size();

	vec4 *baseVertex = &object.m_CurrentVertices.at(vertexOffset);
	vec3 *baseNormal = &object.m_CurrentNormals.at(vertexOffset);
	memcpy(baseVertex, &object.m_BaseVertices.at(vertexOffset), vertexCount * sizeof(glm::vec4));
	memcpy(baseNormal, &object.m_BaseNormals.at(vertexOffset), vertexCount * sizeof(glm::vec3));

	for (uint i = 0; i < vertexCount; i++)
	{
		baseVertex[i] = vec4(object.m_Poses.at(0).positions.at(i), 1.0f);
		for (int j = 1; j <= weightCount; j++)
		{
			baseVertex[i] = baseVertex[i] + (weights[j - 1] * vec4(object.m_Poses.at(j).positions.at(i), 0));
			baseNormal[i] = baseNormal[i] + (weights[j - 1] * object.m_Poses.at(j).normals.at(i));
		}
	}

	const auto offset = vertexOffset / 3;
	for (uint s = (vertexCount / 3), i = 0; i < s; i++)
	{
		auto &tri = object.m_Triangles.at(i + offset);
		tri.vertex0 = vec3(baseVertex[i * 3 + 0]);
		tri.vertex1 = vec3(baseVertex[i * 3 + 1]);
		tri.vertex2 = vec3(baseVertex[i * 3 + 2]);

		tri.vN0 = normalize(baseNormal[i * 3 + 0]);
		tri.vN1 = normalize(baseNormal[i * 3 + 1]);
		tri.vN2 = normalize(baseNormal[i * 3 + 2]);
	}

	object.m_HasUpdated = true;
}

void rfw::gTLFObject::build(const std::vector<int> &indices, const std::vector<glm::vec3> &vertices,
							const std::vector<glm::vec3> &normals, const std::vector<glm::vec2> &uvs,
							const std::vector<gLTFPose> &poses, const std::vector<glm::uvec4> &joints,
							const std::vector<glm::vec4> &weights, const int materialIdx)
{
	using namespace glm;

	// Add mesh information to object
	gLTFMesh mesh;
	mesh.vertexOffset = static_cast<uint>(m_BaseVertices.size());
	mesh.vertexCount = static_cast<uint>(vertices.size());

	mesh.poseOffset = static_cast<uint>(m_Poses.size());
	mesh.poseCount = static_cast<uint>(poses.size());

	mesh.jointCount = static_cast<uint>(joints.size());
	mesh.jointOffset = static_cast<uint>(m_Joints.size());

	mesh.weightCount = static_cast<uint>(weights.size());
	mesh.weightOffset = static_cast<uint>(m_Weights.size());

	m_Meshes.emplace_back(mesh);

	// Allocate data
	const auto triangleOffset = m_Triangles.size();
	const auto vertexOffset = m_BaseVertices.size();
	const auto poseOffset = m_Poses.size();
	const auto jointOffset = m_Joints.size();
	const auto weightOffset = m_Weights.size();

	m_Poses.resize(m_Poses.size() + poses.size());
	m_Triangles.resize(m_Triangles.size() + indices.size() / 3);
	m_BaseVertices.resize(m_BaseVertices.size() + vertices.size());
	m_BaseNormals.resize(m_BaseNormals.size() + vertices.size());
	m_BaseTexCoords.resize(m_BaseTexCoords.size() + vertices.size());

	for (size_t s = indices.size() / 3, i = 0; i < s; i++)
	{
		auto &tri = m_Triangles.at(i + triangleOffset);

		const auto idx = glm::uvec3(indices.at(i * 3 + 0), indices.at(i * 3 + 1), indices.at(i * 3 + 2));
		const auto &v0 = vertices.at(idx.x);
		const auto &v1 = vertices.at(idx.y);
		const auto &v2 = vertices.at(idx.z);

		const auto &n0 = normals.at(idx.x);
		const auto &n1 = normals.at(idx.y);
		const auto &n2 = normals.at(idx.z);

		m_BaseVertices.at(idx.x + vertexOffset) = vec4(v0, 1.0f);
		m_BaseVertices.at(idx.y + vertexOffset) = vec4(v1, 1.0f);
		m_BaseVertices.at(idx.z + vertexOffset) = vec4(v2, 1.0f);

		m_BaseNormals.at(idx.x + vertexOffset) = n0;
		m_BaseNormals.at(idx.y + vertexOffset) = n1;
		m_BaseNormals.at(idx.z + vertexOffset) = n2;

		m_BaseTexCoords.at(idx.x + vertexOffset) = uvs.at(idx.x);
		m_BaseTexCoords.at(idx.y + vertexOffset) = uvs.at(idx.y);
		m_BaseTexCoords.at(idx.z + vertexOffset) = uvs.at(idx.z);

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

// Tangents are not actually used
#if 1
		if (!uvs.empty())
		{
			tri.u0 = uvs.at(idx.x).x;
			tri.u1 = uvs.at(idx.y).x;
			tri.u2 = uvs.at(idx.z).x;

			tri.v0 = uvs.at(idx.x).y;
			tri.v1 = uvs.at(idx.y).y;
			tri.v2 = uvs.at(idx.z).y;

			const auto uv01 = vec2(tri.u1 - tri.u0, tri.v1 - tri.v0);
			const auto uv02 = vec2(tri.u2 - tri.u0, tri.v2 - tri.v0);
			if (dot(uv01, uv01) == 0 || dot(uv02, uv02) == 0)
			{
				tri.T = normalize(tri.vertex1 - tri.vertex0);
				tri.B = normalize(cross(N, tri.T));
			}
			else
			{
				tri.T = normalize((tri.vertex1 - tri.vertex0) * uv02.y - (tri.vertex2 - tri.vertex0) * uv01.y);
				tri.B = normalize((tri.vertex2 - tri.vertex0) * uv01.x - (tri.vertex1 - tri.vertex0) * uv02.x);
			}
		}
		else
		{
			tri.T = normalize(tri.vertex1 - tri.vertex0);
			tri.B = normalize(cross(N, tri.T));
		}
#endif

		tri.material = materialIdx;

		if (!joints.empty())
		{
			m_Joints.at(jointOffset + idx.x) = joints.at(idx.x);
			m_Joints.at(jointOffset + idx.y) = joints.at(idx.y);
			m_Joints.at(jointOffset + idx.z) = joints.at(idx.z);

			m_Weights.at(weightOffset + idx.x) = weights.at(idx.x);
			m_Weights.at(weightOffset + idx.y) = weights.at(idx.y);
			m_Weights.at(weightOffset + idx.z) = weights.at(idx.z);
		}

		for (size_t s = poses.size(), i = 0; i < s; i++)
		{
			m_Poses.at(i + poseOffset).positions.push_back(poses[i].positions.at(idx.x));
			m_Poses.at(i + poseOffset).positions.push_back(poses[i].positions.at(idx.y));
			m_Poses.at(i + poseOffset).positions.push_back(poses[i].positions.at(idx.z));
			m_Poses.at(i + poseOffset).normals.push_back(poses[i].normals.at(idx.x));
			m_Poses.at(i + poseOffset).normals.push_back(poses[i].normals.at(idx.y));
			m_Poses.at(i + poseOffset).normals.push_back(poses[i].normals.at(idx.z));
			m_Poses.at(i + poseOffset).tangents.push_back(poses[i].tangents.at(idx.x));
			m_Poses.at(i + poseOffset).tangents.push_back(poses[i].tangents.at(idx.y));
			m_Poses.at(i + poseOffset).tangents.push_back(poses[i].tangents.at(idx.z));
		}
	}
}

bool rfw::gTLFObject::gLTFNode::update(gTLFObject &object, glm::mat4 &T)
{
	combinedTransform = T * localTransform;
	bool instancesChanged = wasModified;

	for (size_t s = childIndices.size(), i = 0; i < s; i++)
	{
		rfw::gTLFObject::gLTFNode &child = object.m_Nodes.at(childIndices.at(i));
		bool childChanged = child.update(object, combinedTransform);
		instancesChanged |= childChanged;
		treeChanged |= childChanged;
	}

	if (meshID > -1)
	{
		if (morphed)
		{
			object.m_Meshes.at(meshID).setPose(object, weights);
			morphed = false;
		}

		if (wasModified && hasLTris)
			updateLights();

		if (skinID > -1)
		{
			auto &skin = object.m_Skins.at(skinID);
			mat4 meshTransform = combinedTransform;
			mat4 meshTransformInverted = inverse(meshTransform);
			for (size_t s = skin.joints.size(), j = 0; j < s; j++)
			{
				auto &jointNode = object.m_Nodes.at(skin.joints.at(j));
				skin.jointMatrices.at(j) =
					meshTransformInverted * jointNode.combinedTransform * skin.inverseBindMatrices.at(j);
			}
			object.m_Meshes.at(meshID).setPose(object, skin);
		}
	}

	return instancesChanged;
}

void rfw::gTLFObject::gLTFNode::updateTransform()
{
	const auto T = glm::translate(glm::mat4(1.0f), translation);
	const auto R = glm::mat4_cast(rotation);
	const auto S = glm::scale(glm::mat4(1.0f), scale);

	localTransform = T * R * S * matrix;
}

void rfw::gTLFObject::gLTFNode::prepareLights() {}

void rfw::gTLFObject::gLTFNode::updateLights() {}

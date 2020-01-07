#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <tiny_gltf.h>

#include "gLTFObject.h"

#include "../utils/File.h"

#include "../RenderSystem.h"

rfw::SceneAnimation creategLTFAnim(rfw::SceneObject *object, tinygltf::Animation &gltfAnim, tinygltf::Model &gltfModel, int nodeBase);

rfw::MeshSkin convertSkin(const tinygltf::Skin &skin, const tinygltf::Model &model)
{
	rfw::MeshSkin s = {};
	s.name = skin.name;
	s.jointNodes.reserve(skin.joints.size());
	for (auto joint : skin.joints)
		s.jointNodes.emplace_back(joint);

	if (skin.inverseBindMatrices > -1)
	{
		const auto &accessor = model.accessors.at(skin.inverseBindMatrices);
		const auto &bufferView = model.bufferViews.at(accessor.bufferView);
		const auto &buffer = model.buffers.at(bufferView.buffer);

		s.inverseBindMatrices.resize(accessor.count);
		memcpy(s.inverseBindMatrices.data(), &buffer.data.at(accessor.byteOffset + bufferView.byteOffset), accessor.count * sizeof(glm::mat4));

		s.jointMatrices.resize(accessor.count, glm::mat4(1.0f));
	}

	return s;
}

rfw::SceneNode createNode(rfw::gLTFObject &object, const tinygltf::Node &node, const std::vector<std::vector<rfw::TmpPrim>> &meshes)
{
	rfw::SceneNode::Transform T = {};
	glm::mat4 transform = mat4(1.0f);

	if (node.matrix.size() == 16)
	{
		glm::dmat4 T;
		memcpy(value_ptr(T), node.matrix.data(), sizeof(glm::dmat4));
		transform = T;
	}

	if (node.translation.size() == 3)
		T.translation = vec3(static_cast<float>(node.translation[0]), static_cast<float>(node.translation[1]), static_cast<float>(node.translation[2]));

	if (node.rotation.size() == 4)
		T.rotation = quat(static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]), static_cast<float>(node.rotation[1]),
						  static_cast<float>(node.rotation[2]));

	if (node.scale.size() == 3)
		T.scale = vec3(static_cast<float>(node.scale[0]), static_cast<float>(node.scale[1]), static_cast<float>(node.scale[2]));

	auto n = rfw::SceneNode(&object.scene, node.name, node.children, {node.mesh}, {node.skin}, meshes, T, transform);
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

	m_BaseMaterialIdx = static_cast<uint>(matList->getMaterials().size());
	const auto baseTextureIdx = matList->getTextures().size();

	for (const auto &tinyMat : model.materials)
	{
		// https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#materials
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
			{
				if (value.second.has_number_value)
					mat.metallic = (float)value.second.number_value;
			}
			if (value.first == "roughnessFactor")
			{
				if (value.second.has_number_value)
					mat.roughness = (float)value.second.number_value;
			}
			if (value.first == "baseColorTexture")
			{
				for (auto &item : value.second.json_double_value)
				{
					if (item.first == "index")
						mat.map[TEXTURE0].textureID = static_cast<int>(item.second) + static_cast<int>(baseTextureIdx);
					if (item.first == "scale")
						mat.map[TEXTURE0].uvscale = vec2(static_cast<float>(item.second));
					if (item.first == "offset")
						mat.map[TEXTURE0].uvoffset = vec2(static_cast<float>(item.second));
				}
			}
			if (value.first == "normalTexture")
			{
				mat.setFlag(HasNormalMap);
				for (auto &item : value.second.json_double_value)
				{
					if (item.first == "index")
						mat.map[NORMALMAP0].textureID = static_cast<int>(item.second) + static_cast<int>(baseTextureIdx);
					if (item.first == "scale")
						mat.map[NORMALMAP0].uvscale = vec2(static_cast<float>(item.second));
					if (item.first == "offset")
						mat.map[NORMALMAP0].uvoffset = vec2(static_cast<float>(item.second));
				}
			}
			if (value.first == "emissiveFactor")
			{
				if (value.second.has_number_value)
				{
					tinygltf::Parameter p = value.second;
					mat.color = vec3(1) + vec3(p.number_array[0], p.number_array[1], p.number_array[2]);
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

	std::vector<std::vector<TmpPrim>> meshes(model.meshes.size());
	for (size_t i = 0; i < model.meshes.size(); i++)
	{
		const auto &mesh = model.meshes.at(i);

		meshes.at(i).resize(mesh.primitives.size());

		for (size_t s = mesh.primitives.size(), j = 0; j < s; j++)
		{
			const Primitive &prim = mesh.primitives.at(j);

			const Accessor &accessor = model.accessors.at(prim.indices);
			const BufferView &view = model.bufferViews.at(accessor.bufferView);
			const Buffer &buffer = model.buffers.at(view.buffer);
			const unsigned char *a = buffer.data.data() + view.byteOffset + accessor.byteOffset;
			const int byteStride = accessor.ByteStride(view);
			const size_t count = accessor.count;

			std::vector<int> &tmpIndices = meshes[i][j].indices;
			std::vector<glm::vec3> &tmpNormals = meshes[i][j].normals;
			std::vector<glm::vec3> &tmpVertices = meshes[i][j].vertices;
			std::vector<glm::vec2> &tmpUvs = meshes[i][j].uvs;
			std::vector<glm::uvec4> &tmpJoints = meshes[i][j].joints;
			std::vector<glm::vec4> &tmpWeights = meshes[i][j].weights;
			std::vector<rfw::SceneMesh::Pose> &tmpPoses = meshes[i][j].poses;

			meshes[i][j].matID = prim.material > -1 ? prim.material + m_BaseMaterialIdx : 0;

			switch (accessor.componentType)
			{
			case (TINYGLTF_COMPONENT_TYPE_BYTE):
				for (int k = 0; k < count; k++, a += byteStride)
				{
					char value = 0;
					memcpy(&value, a, sizeof(char));
					tmpIndices.push_back(value);
				}
				break;
			case (TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE):
				for (int k = 0; k < count; k++, a += byteStride)
				{
					unsigned char value = 0;
					memcpy(&value, a, sizeof(unsigned char));
					tmpIndices.push_back(value);
				}
				break;
			case (TINYGLTF_COMPONENT_TYPE_SHORT):
				for (int k = 0; k < count; k++, a += byteStride)
				{
					short value = 0;
					memcpy(&value, a, sizeof(short));
					tmpIndices.push_back(value);
				}
				break;
			case (TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT):
				for (int k = 0; k < count; k++, a += byteStride)
				{
					unsigned short value = 0;
					memcpy(&value, a, sizeof(unsigned short));
					tmpIndices.push_back(value);
				}
				break;
			case (TINYGLTF_COMPONENT_TYPE_INT):
				for (int k = 0; k < count; k++, a += byteStride)
				{
					int value = 0;
					memcpy(&value, a, sizeof(int));
					tmpIndices.push_back(value);
				}
				break;
			case (TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT):
				for (int k = 0; k < count; k++, a += byteStride)
				{
					unsigned int value = 0;
					memcpy(&value, a, sizeof(unsigned int));
					tmpIndices.push_back(value);
				}
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
								glm::vec3 value;
								memcpy(value_ptr(value), a, sizeof(vec3));
								tmpVertices.push_back(value);
							}
						}
						else if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE)
						{
							// WARNING("%s", "Double precision positions are not supported (yet).");
							for (size_t p = 0; p < count; p++, a += byteStride)
							{
								glm::dvec3 value;
								memcpy(value_ptr(value), a, sizeof(dvec3));
								tmpVertices.emplace_back(value);
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
								glm::vec3 value;
								memcpy(value_ptr(value), a, sizeof(glm::vec3));
								tmpNormals.push_back(value);
							}
						}
						else if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE)
						{
							// WARNING("%s", "Double precision positions are not supported (yet).");
							for (size_t f = 0; f < count; f++, a += byteStride)
							{
								glm::dvec3 value;
								memcpy(value_ptr(value), a, sizeof(glm::dvec3));
								tmpNormals.push_back(value);
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
								glm::vec2 value;
								memcpy(value_ptr(value), a, sizeof(glm::vec2));
								tmpUvs.push_back(value);
							}
						}
						else if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE)
						{
							// WARNING("%s", "Double precision normals are not supported (yet).");
							for (size_t f = 0; f < count; f++, a += byteStride)
							{
								glm::dvec2 value;
								memcpy(value_ptr(value), a, sizeof(glm::dvec2));
								tmpUvs.emplace_back(value);
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
								const double norm = 1.0 / (w4.x + w4.y + w4.z + w4.w);
								w4 *= norm;
								tmpWeights.emplace_back(vec4(w4));
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
		}
	}

	const bool hasTransform = matrix != glm::mat4(1.0f);

	if (model.scenes.size() > 1)
		WARNING("gLTF files with more than 1 scene are not supported (yet).");

	scene.nodes.reserve(model.nodes.size() + 1);
	tinygltf::Scene &gltfScene = model.scenes.at(0);

	for (size_t s = model.nodes.size(), i = 0; i < s; i++)
	{
		const auto &node = model.nodes.at(i);
		scene.nodes.emplace_back(createNode(*this, node, meshes));
	}

	for (int i = 0, s = static_cast<int>(gltfScene.nodes.size()); i < s; i++)
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

	scene.updateTriangles();
	//// Update triangle data that only has to be calculated once
	scene.updateTriangles(matList);

	DEBUG("Loaded file: %s with %u vertices and %u triangles", filename.data(), scene.vertices.size(), scene.triangles.size());
}

void rfw::gLTFObject::transformTo(float timeInSeconds) { scene.transformTo(timeInSeconds); }

rfw::Triangle *rfw::gLTFObject::getTriangles() { return scene.triangles.data(); }

glm::vec4 *rfw::gLTFObject::getVertices() { return scene.vertices.data(); }

const std::vector<std::pair<size_t, rfw::Mesh>> &rfw::gLTFObject::getMeshes() const { return m_Meshes; }

const std::vector<rfw::simd::matrix4> &rfw::gLTFObject::getMeshTransforms() const { return scene.meshTranforms; }

std::vector<bool> rfw::gLTFObject::getChangedMeshes()
{
	auto changed = std::vector<bool>(m_Meshes.size(), false);
	for (int i = 0, s = static_cast<int>(m_Meshes.size()); i < s; i++)
	{
		if (scene.meshes[i].dirty)
		{
			changed[i] = true;
			scene.meshes[i].dirty = false;
		}
	}

	return changed;
}

std::vector<bool> rfw::gLTFObject::getChangedMeshMatrices()
{
	auto values = std::move(scene.changedMeshNodeTransforms);
	scene.changedMeshNodeTransforms.resize(m_Meshes.size(), false);
	return values;
}

bool rfw::gLTFObject::isAnimated() const { return !scene.animations.empty(); }

const std::vector<std::vector<int>> &rfw::gLTFObject::getLightIndices(const std::vector<bool> &matLightFlags, bool reinitialize)
{
	if (reinitialize)
	{
		m_LightIndices.clear();
		m_LightIndices.resize(scene.meshes.size());

		for (size_t i = 0, s = scene.meshes.size(); i < s; i++)
		{
			auto &currentLightVector = m_LightIndices[i];

			const auto &mesh = scene.meshes[i];
			const Triangle *triangles = scene.meshes[i].getTriangles();

			for (int t = 0, st = static_cast<int>(mesh.faceCount); t < st; t++)
			{
				if (matLightFlags[triangles[t].material])
					currentLightVector.push_back(t);
			}
		}
	}

	return m_LightIndices;
}

void rfw::gLTFObject::prepareMeshes(RenderSystem &rs)
{
	m_Meshes.clear();
#if 1
	for (const auto &mesh : scene.meshes)
	{
		auto m = rfw::Mesh();
		m.vertices = mesh.getVertices();
		m.normals = mesh.getNormals();
		m.vertexCount = mesh.vertexCount;
		m.triangleCount = mesh.faceCount;
		m.triangles = mesh.getTriangles();
		m.indices = mesh.getIndices();
		m.texCoords = mesh.getTexCoords();
		m_Meshes.emplace_back(rs.requestMeshIndex(), m);
	}
#else
	mesh.vertices = scene.vertices.data();
	mesh.normals = scene.normals.data();
	mesh.triangles = scene.triangles.data();
	if (scene.indices.empty())
	{
		mesh.indices = nullptr;
		mesh.triangleCount = scene.vertices.size() / 3;
	}
	else
	{
		mesh.indices = scene.indices.data();
		mesh.triangleCount = scene.indices.size();
	}
	mesh.vertexCount = scene.vertices.size();
	mesh.texCoords = scene.texCoords.data();
	m_Meshes.emplace_back(rs.requestMeshIndex(), mesh);
#endif
}

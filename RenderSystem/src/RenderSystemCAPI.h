//
// Created by Mèir Noordermeer on 29/10/2019.
//

#ifndef RENDERINGFW_RENDERSYSTEM_SRC_RENDERSYSTEMC_H
#define RENDERINGFW_RENDERSYSTEM_SRC_RENDERSYSTEMC_H

#include <GLFW/glfw3.h>

struct rfwvec2
{
	float x, y;
};

struct rfwvec3
{
	float x, y, z;
};

struct rfwvec4
{
	float x, y, z, w;
};

struct rfwivec2
{
	int x, y;
};

struct rfwivec3
{
	int x, y, z;
};

struct rfwivec4
{
	int x, y, z, w;
};

struct rfwuvec2
{
	unsigned int x, y;
};

struct rfwuvec3
{
	unsigned int x, y, z;
};

struct uvec4
{
	unsigned int x, y, z, w;
};

struct rfwAABB
{
	rfwvec3 mMin;
	rfwvec3 mMax;
};

struct rfwProbeResult
{
	unsigned int instanceIdx;
	float distance;
	unsigned int materialIdx;
};

struct rfwRenderSystemHandle
{
	void *instance;
};

struct rfwInstanceReference
{
	unsigned int index;
};

struct rfwGeometryReference
{
	unsigned int index;
};

struct MapProps
{
	int textureID;	  // texture ID; -1 denotes empty slot
	float valueScale; // texture value scale, only sensible for normal maps
	rfwvec2 uvscale;  // uv coordinate scale
	rfwvec2 uvoffset; // uv coordinate offset
};

struct Camera
{
	// data members
	rfwvec3 position;  // position of the centre of the lens
	rfwvec3 direction; // camera target

	float focalDistance = 5.0f; // distance of the focal plane
	float aperture = 0.0001f;	// aperture size
	float brightness = 0.0f;	// combined with contrast:
	float contrast = 0.0f;		// pragmatic representation of exposure
	float FOV = 40.0f;			// field of view, in degrees
	float aspectRatio = 1.0f;	// image plane aspect ratio
	float clampValue = 10.0f;	// firefly clamping
	rfwivec2 pixelCount;		// actual pixel count; needed for pixel spread angle
};

struct HostMaterial
{
	enum
	{
		SMOOTH = 1,		   // material uses normal interpolation
		HASALPHA = 2,	   // material textures use alpha channel
		ANISOTROPIC = 4,   // material has anisotropic roughness
		FROM_MTL = 128,	   // changes are persistent for these, not for others
		ISCONDUCTOR = 256, // rough conductor
		ISDIELECTRIC = 512 // rough dielectric. If 256 and 512 not specified: diffuse.
	};

	const char *name;			 // material name, not for unique identification
	const char *origin;			 // origin: file from which the data was loaded, with full path
	int ID;						 // unique integer ID of this material
	unsigned int flags = SMOOTH; // material properties
	unsigned int refCount = 1;	 // the number of models that use this material

	rfwvec3 color;
	rfwvec3 absorption;
	float metallic;
	float subsurface;
	float specular;
	float roughness;
	float specularTint;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatGloss;
	float transmission;
	float eta;
	float custom0;
	float custom1;
	float custom2;
	float custom3;
	MapProps map[11]; // bitmap data
	// field for the BuildMaterialList method of HostMesh
	bool visited = false; // last mesh that checked this material
};

#define LIGHT_TYPE_AREA 0
#define LIGHT_TYPE_POINT 1
#define LIGHT_TYPE_SPOT 2
#define LIGHT_TYPE_DIRECTIONAL 3

struct LightReference
{
	unsigned int index;
	unsigned int type;
};

struct AvailableRenderSettings
{
	unsigned int numKeys;
	char **settingKeys;
	unsigned int numSettingValues;
	char **settingValues;
};

struct RenderSetting
{
	const char *name;
	const char *value;
};

rfwRenderSystemHandle createRenderSystem();
void cleanupRenderSystem(rfwRenderSystemHandle *handle);
void loadRenderAPI(rfwRenderSystemHandle *handle, const char *name);
void unloadRenderAPI(rfwRenderSystemHandle *handle);
void setTarget(rfwRenderSystemHandle *handle, GLuint *textureID, unsigned int width, unsigned int height);
void setSkybox(rfwRenderSystemHandle *handle, const char *filename);
void synchronize(rfwRenderSystemHandle *handle);
void updateAnimationsTo(rfwRenderSystemHandle *handle, float timeInSeconds);
rfwGeometryReference addObject(rfwRenderSystemHandle *handle, const char *fileName, int material = -1);
rfwGeometryReference addObject(rfwRenderSystemHandle *handle, const char *fileName, bool normalize, int material = -1);
rfwGeometryReference addObject(rfwRenderSystemHandle *handle, const char *fileName, bool normalize,
							   const float *preTransform = nullptr, int material = -1);
rfwGeometryReference addQuad(rfwRenderSystemHandle *handle, rfwvec3 N, rfwvec3 pos, float width, float height,
							 unsigned int material);
rfwInstanceReference addInstance(rfwRenderSystemHandle *handle, rfwGeometryReference geometry, const rfwvec3 &scaling,
								 const rfwvec3 &translation, float degrees, const rfwvec3 &axes);
void updateInstance(rfwRenderSystemHandle *handle, rfwInstanceReference instanceRef, const float *transform = NULL);
void setAnimationTime(rfwRenderSystemHandle *handle, rfwGeometryReference geometryRef, float timeInSeconds);
HostMaterial getMaterial(rfwRenderSystemHandle *handle, size_t index);
void setMaterial(rfwRenderSystemHandle *handle, size_t index, const HostMaterial &mat);
unsigned int addMaterial(rfwRenderSystemHandle *handle, const rfwvec3 &color, float roughness = 1.0f);

#define RESET 0
#define CONVERGE 1

void renderFrame(rfwRenderSystemHandle *handle, const Camera &camera, unsigned int status = CONVERGE);

LightReference addPointLight(rfwRenderSystemHandle *handle, const rfwvec3 &position, float energy,
							 const rfwvec3 &radiance);
LightReference addSpotLight(rfwRenderSystemHandle *handle, const rfwvec3 &position, float cosInner,
							const rfwvec3 &radiance, float cosOuter, float energy, const rfwvec3 &direction);
LightReference addDirectionalLight(rfwRenderSystemHandle *handle, const rfwvec3 &direction, float energy,
								   const rfwvec3 &radiance);

void setPosition(rfwRenderSystemHandle *handle, const LightReference &reference, const rfwvec3 &position);
void setRadiance(rfwRenderSystemHandle *handle, const LightReference &reference, const rfwvec3 &radiance);
void setEnergy(rfwRenderSystemHandle *handle, const LightReference &reference, float energy);
AvailableRenderSettings getAvailableSettings(rfwRenderSystemHandle *handle);
void setSetting(rfwRenderSystemHandle *handle, const RenderSetting &setting);
rfwAABB calculateSceneBounds(rfwRenderSystemHandle *handle);

void setProbeIndex(rfwRenderSystemHandle *handle, rfwuvec2 pixelIdx);
rfwProbeResult getProbeResult(rfwRenderSystemHandle *handle);

#endif // RENDERINGFW_RENDERSYSTEM_SRC_RENDERSYSTEMC_H

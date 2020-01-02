//
// Created by meir on 10/25/19.
//

#ifndef RENDERINGFW_VULKANRTX_SRC_CHECKVK_H
#define RENDERINGFW_VULKANRTX_SRC_CHECKVK_H

#define CheckVK(x) _CheckVK(__LINE__, __FILE__, static_cast<vk::Result>(x))
static void _CheckVK(int line, const char *file, vk::Result x)
{
	const char *error;

	switch (x)
	{
	case (vk::Result::eSuccess):
		return;
	case (vk::Result::eNotReady):
		error = "VK_NOT_READY";
		break;
	case (vk::Result::eTimeout):
		error = "VK_TIMEOUT";
		break;
	case (vk::Result::eEventSet):
		error = "VK_EVENT_SET";
		break;
	case (vk::Result::eIncomplete):
		error = "VK_INCOMPLETE";
		break;
	case (vk::Result::eErrorOutOfHostMemory):
		error = "VK_ERROR_OUT_OF_HOST_MEMORY";
		break;
	case (vk::Result::eErrorOutOfDeviceMemory):
		error = "VK_ERROR_OUT_OF_DEVICE_MEMORY";
		break;
	case (vk::Result::eErrorInitializationFailed):
		error = "VK_ERROR_INITIALIZATION_FAILED";
		break;
	case (vk::Result::eErrorDeviceLost):
		error = "VK_ERROR_DEVICE_LOST";
		break;
	case (vk::Result::eErrorMemoryMapFailed):
		error = "VK_ERROR_MEMORY_MAP_FAILED";
		break;
	case (vk::Result::eErrorLayerNotPresent):
		error = "VK_ERROR_LAYER_NOT_PRESENT";
		break;
	case (vk::Result::eErrorExtensionNotPresent):
		error = "VK_ERROR_EXTENSION_NOT_PRESENT";
		break;
	case (vk::Result::eErrorFeatureNotPresent):
		error = "VK_ERROR_FEATURE_NOT_PRESENT";
		break;
	case (vk::Result::eErrorIncompatibleDriver):
		error = "VK_ERROR_INCOMPATIBLE_DRIVER";
		break;
	case (vk::Result::eErrorTooManyObjects):
		error = "VK_ERROR_TOO_MANY_OBJECTS";
		break;
	case (vk::Result::eErrorFormatNotSupported):
		error = "VK_ERROR_FORMAT_NOT_SUPPORTED";
		break;
	case (vk::Result::eErrorFragmentedPool):
		error = "VK_ERROR_FRAGMENTED_POOL";
		break;
	case (vk::Result::eErrorOutOfPoolMemory):
		error = "VK_ERROR_OUT_OF_POOL_MEMORY";
		break;
	case (vk::Result::eErrorInvalidExternalHandle):
		error = "VK_ERROR_INVALID_EXTERNAL_HANDLE";
		break;
	case (vk::Result::eErrorSurfaceLostKHR):
		error = "VK_ERROR_SURFACE_LOST_KHR";
		break;
	case (vk::Result::eErrorNativeWindowInUseKHR):
		error = "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
		break;
	case (vk::Result::eSuboptimalKHR):
		error = "VK_SUBOPTIMAL_KHR";
		break;
	case (vk::Result::eErrorOutOfDateKHR):
		error = "VK_ERROR_OUT_OF_DATE_KHR";
		break;
	case (vk::Result::eErrorIncompatibleDisplayKHR):
		error = "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
		break;
	case (vk::Result::eErrorValidationFailedEXT):
		error = "VK_ERROR_VALIDATION_FAILED_EXT";
		break;
	case (vk::Result::eErrorInvalidShaderNV):
		error = "VK_ERROR_INVALID_SHADER_NV";
		break;
	case (vk::Result::eErrorInvalidDrmFormatModifierPlaneLayoutEXT):
		error = "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
		break;
	case (vk::Result::eErrorFragmentationEXT):
		error = "VK_ERROR_FRAGMENTATION_EXT";
		break;
	case (vk::Result::eErrorNotPermittedEXT):
		error = "VK_ERROR_NOT_PERMITTED_EXT";
		break;
	case (vk::Result::eErrorInvalidDeviceAddressEXT):
		error = "VK_ERROR_INVALID_DEVICE_ADDRESS_EXT";
		break;
	case (vk::Result::eErrorFullScreenExclusiveModeLostEXT):
		error = "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
		break;
	default:
		error = "UNKNOWN";
		break;
	}

	FAILURE("Vulkan error on line %d in %s: %u = %s", line, file, uint(x), error);
}

#endif // RENDERINGFW_VULKANRTX_SRC_CHECKVK_H

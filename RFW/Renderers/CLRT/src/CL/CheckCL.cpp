#include "../PCH.h"

namespace cl
{
bool _CheckCL(cl_int result, const char *file, int line)
{
	switch (result)
	{
	case (CL_SUCCESS):
		return true;
	case (CL_DEVICE_NOT_FOUND):
		rfw::utils::logger::_warning(file, line, "Error: CL_DEVICE_NOT_FOUND");
		break;
	case (CL_DEVICE_NOT_AVAILABLE):
		rfw::utils::logger::_warning(file, line, "Error: CL_DEVICE_NOT_AVAILABLE");
		break;
	case (CL_COMPILER_NOT_AVAILABLE):
		rfw::utils::logger::_warning(file, line, "Error: CL_COMPILER_NOT_AVAILABLE");
		break;
	case (CL_OUT_OF_RESOURCES):
		rfw::utils::logger::_warning(file, line, "Error: CL_OUT_OF_RESOURCES");
		break;
	case (CL_OUT_OF_HOST_MEMORY):
		rfw::utils::logger::_warning(file, line, "Error: CL_OUT_OF_HOST_MEMORY");
		break;
	case (CL_PROFILING_INFO_NOT_AVAILABLE):
		rfw::utils::logger::_warning(file, line, "Error: CL_PROFILING_INFO_NOT_AVAILABLE");
		break;
	case (CL_MEM_COPY_OVERLAP):
		rfw::utils::logger::_warning(file, line, "Error: CL_MEM_COPY_OVERLAP");
		break;
	case (CL_IMAGE_FORMAT_MISMATCH):
		rfw::utils::logger::_warning(file, line, "Error: CL_IMAGE_FORMAT_MISMATCH");
		break;
	case (CL_IMAGE_FORMAT_NOT_SUPPORTED):
		rfw::utils::logger::_warning(file, line, "Error: CL_IMAGE_FORMAT_NOT_SUPPORTED");
		break;
	case (CL_MISALIGNED_SUB_BUFFER_OFFSET):
		rfw::utils::logger::_warning(file, line, "Error: CL_MISALIGNED_SUB_BUFFER_OFFSET");
		break;
	case (CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST):
		rfw::utils::logger::_warning(file, line, "Error: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST");
		break;
	case (CL_INVALID_VALUE):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_VALUE");
		break;
	case (CL_INVALID_DEVICE_TYPE):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_DEVICE_TYPE");
		break;
	case (CL_INVALID_PLATFORM):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_PLATFORM");
		break;
	case (CL_INVALID_DEVICE):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_DEVICE");
		break;
	case (CL_INVALID_CONTEXT):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_CONTEXT");
		break;
	case (CL_INVALID_QUEUE_PROPERTIES):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_QUEUE_PROPERTIES");
		break;
	case (CL_INVALID_COMMAND_QUEUE):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_COMMAND_QUEUE");
		break;
	case (CL_INVALID_HOST_PTR):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_HOST_PTR");
		break;
	case (CL_INVALID_MEM_OBJECT):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_MEM_OBJECT");
		break;
	case (CL_INVALID_IMAGE_FORMAT_DESCRIPTOR):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_IMAGE_FORMAT_DESCRIPTOR");
		break;
	case (CL_INVALID_IMAGE_SIZE):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_IMAGE_SIZE");
		break;
	case (CL_INVALID_SAMPLER):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_SAMPLER");
		break;
	case (CL_INVALID_BINARY):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_BINARY");
		break;
	case (CL_INVALID_BUILD_OPTIONS):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_BUILD_OPTIONS");
		break;
	case (CL_INVALID_PROGRAM):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_PROGRAM");
		break;
	case (CL_INVALID_PROGRAM_EXECUTABLE):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_PROGRAM_EXECUTABLE");
		break;
	case (CL_INVALID_KERNEL_NAME):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_KERNEL_NAME");
		break;
	case (CL_INVALID_KERNEL_DEFINITION):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_KERNEL_DEFINITION");
		break;
	case (CL_INVALID_KERNEL):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_KERNEL");
		break;
	case (CL_INVALID_ARG_INDEX):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_ARG_INDEX");
		break;
	case (CL_INVALID_ARG_VALUE):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_ARG_VALUE");
		break;
	case (CL_INVALID_ARG_SIZE):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_ARG_SIZE");
		break;
	case (CL_INVALID_KERNEL_ARGS):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_KERNEL_ARGS");
		break;
	case (CL_INVALID_WORK_DIMENSION):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_WORK_DIMENSION");
		break;
	case (CL_INVALID_WORK_GROUP_SIZE):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_WORK_GROUP_SIZE");
		break;
	case (CL_INVALID_WORK_ITEM_SIZE):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_WORK_ITEM_SIZE");
		break;
	case (CL_INVALID_GLOBAL_OFFSET):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_GLOBAL_OFFSET");
		break;
	case (CL_INVALID_EVENT_WAIT_LIST):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_EVENT_WAIT_LIST");
		break;
	case (CL_INVALID_EVENT):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_EVENT");
		break;
	case (CL_INVALID_OPERATION):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_OPERATION");
		break;
	case (CL_INVALID_GL_OBJECT):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_GL_OBJECT");
		break;
	case (CL_INVALID_BUFFER_SIZE):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_BUFFER_SIZE");
		break;
	case (CL_INVALID_MIP_LEVEL):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_MIP_LEVEL");
		break;
	case (CL_INVALID_GLOBAL_WORK_SIZE):
		rfw::utils::logger::_warning(file, line, "Error: CL_INVALID_GLOBAL_WORK_SIZE");
		break;
	default:
		return false;
	}

	return false;
}
} // namespace cl
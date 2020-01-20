#include "../PCH.h"

namespace cl
{

// source file information
static int sourceFiles = 0;
static std::string sourceFile[64]; // yup, ugly constant

static std::map<std::string, std::string> source_files;

using namespace glm;

static cl_device_id getFirstDevice(cl_context context)
{
	size_t dataSize;
	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &dataSize);
	std::vector<cl_device_id> devices(dataSize);
	clGetContextInfo(context, CL_CONTEXT_DEVICES, dataSize, devices.data(), nullptr);
	cl_device_id first = devices[0];
	return first;
}

static cl_int getPlatformID(cl_platform_id *platform)
{
	cl_uint num_platforms, devCount;

	CheckCL(clGetPlatformIDs(0, nullptr, &num_platforms));
	if (num_platforms == 0)
	{
		FAILURE("No valid OpenCL platforms found.");
	}

	std::vector<cl_platform_id> clPlatformIDs(num_platforms);
	CheckCL(clGetPlatformIDs(num_platforms, clPlatformIDs.data(), nullptr));

	const std::array<cl_uint, 2> deviceTypes = {CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
	const std::array<std::string, 4> deviceOrder = {"", "INTEL", "AMD", "NVIDIA"};

	rfw::utils::logger::log("Available OpenCL platforms:");
	for (cl_uint i = 0; i < num_platforms; ++i)
	{
		char platform_name[512];
		CheckCL(clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, sizeof(platform_name), &platform_name, nullptr));
		rfw::utils::logger::log("Platform (%i): %s", i, platform_name);
	}

	for (const auto dev_type : deviceTypes)
	{
		for (const auto &preferred_device : deviceOrder)
		{
			for (cl_uint i = 0; i < num_platforms; ++i)
			{
				auto error = clGetDeviceIDs(clPlatformIDs[i], dev_type, 0, nullptr, &devCount);
				if ((error != CL_SUCCESS) || (devCount == 0))
					continue;

				char buffer[512];

				CheckCL(clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, sizeof(buffer), &buffer, nullptr));
				std::transform(buffer, buffer + sizeof(buffer), buffer, ::toupper);

				if (!preferred_device.empty() && !strstr(buffer, preferred_device.data()))
					continue;

				rfw::utils::logger::log("Current platform: %s", buffer);
				*platform = clPlatformIDs[i];
				break;
			}
		}
	}

	assert(*platform != nullptr);
	return ((*platform) != nullptr) ? CL_SUCCESS : CL_INVALID_PLATFORM;
}

std::vector<std::string> split_path(const std::string &str, const std::set<char> delimiters)
{
	std::vector<std::string> result;

	char const *pch = str.c_str();
	char const *start = pch;
	for (; *pch; ++pch)
	{
		if (delimiters.find(*pch) != delimiters.end())
		{
			if (start != pch)
			{
				std::string str(start, pch);
				result.push_back(str);
			}
			else
				result.emplace_back("");
			start = pch + 1;
		}
	}
	result.emplace_back(start);

	return result;
}

static std::string loadSource(const char *file, size_t *size)
{
	std::string source;
	// extract path from source file name
	char path[2048];
	strcpy(path, file);
	char *marker = path;
	char *fileName = const_cast<char *>(file);

	while (strstr(marker + 1, "\\"))
		marker = strstr(marker + 1, "\\");
	while (strstr(marker + 1, "/"))
		marker = strstr(marker + 1, "/");
	while (strstr(fileName + 1, "\\"))
		fileName = strstr(fileName + 1, "\\");
	while (strstr(fileName + 1, "/"))
		fileName = strstr(fileName + 1, "/");
	if (fileName != file)
		fileName++;
	sourceFile[sourceFiles] = std::string();
	sourceFile[sourceFiles] = fileName;
	*marker = 0;

	// load source file
	FILE *f = fopen(file, "r");
	if (!f)
		FAILURE("Could not load file: %s", file);
	char line[8192];
	int lineNr = 0, currentFile = ((sourceFiles + 1) % 64);
	while (!feof(f))
	{
		line[0] = 0;
		fgets(line, 8190, f);
		lineNr++;
		// clear source file line
		while (line[0])
		{
			if (line[strlen(line) - 1] > 32)
				break;
			line[strlen(line) - 1] = 0;
		}

		// expand error commands
		char *err = strstr(line, "Error(");
		if (err)
		{
			char rem[8192], cmd[128];
			strcpy(rem, err + 6);
			*err = 0;
			sprintf(cmd, "Error_( %i, %i,", currentFile, lineNr);
			strcat(line, cmd);
			strcat(line, rem);
		}

		// expand assets
		char *as = strstr(line, "Assert(");
		if (as)
		{
			char rem[8192], cmd[128];
			strcpy(rem, as + 7);
			*as = 0;
			sprintf(cmd, "Assert_( %i, %i,", currentFile, lineNr);
			strcat(line, cmd);
			strcat(line, rem);
		}

		// handle include files
		char *inc = strstr(line, "#include");
		if (inc)
		{
			char *start = strstr(inc, "\"");
			if (!start)
				FAILURE("Preprocessor error in #include statement line");
			char *end = strstr(start + 1, "\"");
			if (!end)
				FAILURE("Preprocessor error in #include statement line");
			char included_file[2048];
			*end = 0;
			strcpy(included_file, path);
			strcat(included_file, "/");
			strcat(included_file, start + 1);
			if (!rfw::utils::file::exists(file))
				FAILURE("Included file (%s) of (%s) could not be found.", fileName, included_file);

			const auto incText = loadSource(included_file, size);
			source.append(incText.data());
		}
		else
		{
			source.append(line);
			source.append("\n");
		}
	}

	*size = strlen(source.c_str());
	return source;
}

CLContext::CLContext()
{
	if (!init_cl())
		FAILURE("Could not initialize OpenCL context.");
}

CLContext::~CLContext()
{

	if (m_Queue)
		clReleaseCommandQueue(m_Queue);
	if (m_Device)
		clReleaseDevice(m_Device);
	if (m_Context)
		clReleaseContext(m_Context);
}

bool CLContext::init_cl()
{
	cl_platform_id platform = nullptr;
	cl_uint devCount = 0;
	cl_int error = 0;

	if (!CheckCL(error = getPlatformID(&platform)))
		return false;
#ifdef __APPLE__
	if (!CheckCL(error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &devCount)))
		return false;
	std::vector<cl_device_id> devices(devCount);
	if (!CheckCL(error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, devCount, devices.data(), nullptr)))
		return false;

	if (devCount >= 2)
	{
		// swap devices so dGPU gets selected
		std::swap(devices[0], devices[1]);
	}
#else
	if (!CheckCL(error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devCount)))
		return false;
	std::vector<cl_device_id> devices(devCount);
	if (!CheckCL(error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devCount, devices.data(), nullptr)))
		return false;
#endif
	uint deviceUsed = 0;
	uint endDev = devCount;
	bool canShare = false;

	for (uint i = deviceUsed; (!canShare && (i < endDev)); ++i)
	{
		size_t extensionSize;
		CheckCL(error = clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, 0, nullptr, &extensionSize));
		if (extensionSize > 0)
		{
			std::vector<char> extension_string(extensionSize);
			CheckCL(error = clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, extensionSize, extension_string.data(),
											&extensionSize));
			std::string device_extensions(extension_string.data());
			std::transform(device_extensions.begin(), device_extensions.end(), device_extensions.begin(), ::tolower);
			size_t oldPos = 0;
			size_t spacePos = device_extensions.find(' ', oldPos); // extensions string is space delimited

#ifdef __APPLE__
			const char *neededProp = "cl_apple_gl_sharing";
#else
			const char *neededProp = "cl_khr_gl_sharing";
#endif

			while (spacePos != std::string::npos)
			{
				if (strcmp(neededProp, device_extensions.substr(oldPos, spacePos - oldPos).c_str()) == 0)
				{
					canShare = true; // device supports context sharing with OpenGL
					deviceUsed = i;
					break;
				}
				do
				{
					oldPos = spacePos + 1;
					spacePos = device_extensions.find(' ', oldPos);
				} while (spacePos == oldPos);
			}
		}
	}

	if (canShare)
	{
		rfw::utils::logger::log("Using CL-GL Interop.");
	}
	else
	{
		rfw::utils::logger::log("No device found that supports CL/GL context sharing.");
	}

#ifdef _WIN32
	const cl_context_properties props[] = {CL_GL_CONTEXT_KHR,
										   (cl_context_properties)wglGetCurrentContext(),
										   CL_WGL_HDC_KHR,
										   (cl_context_properties)wglGetCurrentDC(),
										   CL_CONTEXT_PLATFORM,
										   (cl_context_properties)platform,
										   0};
#elif defined(__APPLE__)
	CGLContextObj kCGLContext = CGLGetCurrentContext();
	CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
	const cl_context_properties props[] = {CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
										   (cl_context_properties)kCGLShareGroup, CL_CONTEXT_PLATFORM,
										   (cl_context_properties)platform, 0};
#elif defined(__linux)
	const cl_context_properties props[] = {CL_GL_CONTEXT_KHR,
										   (cl_context_properties)glXGetCurrentContext(),
										   CL_GLX_DISPLAY_KHR,
										   (cl_context_properties)glXGetCurrentDisplay(),
										   CL_CONTEXT_PLATFORM,
										   (cl_context_properties)platform,
										   0};
#else
	static_assert(false, "Unsupported OpenCL platform.");
#endif

	// attempt to create a context with the requested features
	m_CanDoInterop = true;
	m_Context = clCreateContext(props, 1, &devices[deviceUsed], nullptr, nullptr, &error);
	if (error != CL_SUCCESS)
	{
		// that didn't work, let's take what we can get
		cl_context_properties temp_props[] = {0};
		m_Context = clCreateContext(temp_props, 1, &devices[deviceUsed], nullptr, nullptr, &error);
		m_CanDoInterop = false;
	}

	m_Device = getFirstDevice(m_Context);
	if (!CheckCL(error))
		return false;

	// print device name
	char device_string[1024];
	clGetDeviceInfo(devices[deviceUsed], CL_DEVICE_NAME, 1024, &device_string, nullptr);
	printf("Device # %u, %s\n", deviceUsed, device_string);

	size_t p_size;
	size_t l_size;
	size_t d_size;

	clGetDeviceInfo(devices[deviceUsed], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &p_size, nullptr);
	clGetDeviceInfo(devices[deviceUsed], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t), &l_size, nullptr);
	clGetDeviceInfo(devices[deviceUsed], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(size_t), &d_size, nullptr);

	rfw::utils::logger::log("\tMax Work Group Size: %u", p_size);
	rfw::utils::logger::log("\tMax Work Item Sizes: %u", l_size);
	rfw::utils::logger::log("\tMax Work Item Dimensions: %u", d_size);

	// create a command-queue
	m_Queue = clCreateCommandQueue(m_Context, devices[deviceUsed], 0, &error);

	return CheckCL(error);
}

void CLContext::submit(const CLKernel &k) const
{
	const auto kernel = k.get_kernel();
	const auto dims = k.get_dimensions();
	const auto work_size = k.get_work_size();
	const auto local_size = k.get_local_size();

	CheckCL(clEnqueueNDRangeKernel(m_Queue, kernel, dims, 0, work_size, local_size, 0, 0, 0));
}

void CLContext::finish() const { clFinish(m_Queue); }

CLKernel::CLKernel(std::shared_ptr<CLContext> context, const char *file, const char *entryPoint,
				   std::array<size_t, 3> workSize, std::array<size_t, 3> localSize)
	: m_Context(context), m_WorkSize(workSize), m_LocalSize(localSize)
{
	size_t size;
	cl_int error;
	const auto source = loadSource(file, &size);
	const char *source_ptr = source.data();
	m_Program =
		clCreateProgramWithSource(context->get_context(), 1, static_cast<const char **>(&source_ptr), &size, &error);
	CheckCL(error);
	error = clBuildProgram(m_Program, 0, nullptr,
						   "-cl-fast-relaxed-math -cl-mad-enable "
						   "-cl-denorms-are-zero -cl-no-signed-zeros "
						   "-cl-unsafe-math-optimizations -cl-finite-math-only",
						   nullptr, nullptr);

	if (error != CL_SUCCESS)
	{
		std::vector<char> log(100 * 1024, 0);
		clGetProgramBuildInfo(m_Program, context->get_device(), CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
		WARNING("Could not load kernel: %s", file);
		WARNING(log.data());
		throw std::runtime_error(log.data());
	}
	else
	{
		m_Kernel = clCreateKernel(m_Program, entryPoint, &error);
		CheckCL(error);
	}

	set_local_size(localSize);
	set_work_size(workSize);
}

CLKernel::~CLKernel()
{
	if (m_Kernel)
		clReleaseKernel(m_Kernel);
	if (m_Program)
		clReleaseProgram(m_Program);
}

void CLKernel::run() { m_Context->submit(*this); }

void CLKernel::set_offset(const std::array<size_t, 3> &offset) { m_Offset = offset; }

void CLKernel::set_work_size(const std::array<size_t, 3> &work_size)
{
	for (int i = 0; i < 3; i++)
		m_WorkSize[i] = work_size[i];

	m_Dimensions = 0;
	for (int i = 0; i < 3; i++)
	{
		if (m_WorkSize[i] <= 1)
			break;
		else
			m_Dimensions++;
	}

	assert(m_Dimensions != 0);
}

void CLKernel::set_local_size(const std::array<size_t, 3> &local_size)
{
	for (int i = 0; i < 3; i++)
		m_LocalSize[i] = local_size[i];
}

} // namespace cl
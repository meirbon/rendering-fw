#ifndef RENDERINGFW_RFW_RENDERERS_CLRT_SRC_CL_CHECKCL_H
#define RENDERINGFW_RFW_RENDERERS_CLRT_SRC_CL_CHECKCL_H

namespace cl
{
#define CheckCL(x) _CheckCL(x, __FILE__, __LINE__)
bool _CheckCL(cl_int result, const char *file, int line);
} // namespace cl

#endif // RENDERINGFW_RFW_RENDERERS_CLRT_SRC_CL_CHECKCL_H

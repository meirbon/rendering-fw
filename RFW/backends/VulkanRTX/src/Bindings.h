#ifndef RENDERINGFW_VULKANRTX_SRC_BINDINGS_H
#define RENDERINGFW_VULKANRTX_SRC_BINDINGS_H

// RT bindings
#define rtACCELERATION_STRUCTURE 0
#define rtCAMERA 1
#define rtPATH_STATES 2
#define rtPATH_ORIGINS 3
#define rtPATH_DIRECTIONS 4
#define rtPOTENTIAL_CONTRIBUTIONS 5
#define rtACCUMULATION_BUFFER 6
#define rtBLUENOISE 7

// Shade bindings
#define cCOUNTERS 0
#define cCAMERA 1
#define cPATH_STATES 2
#define cPATH_ORIGINS 3
#define cPATH_DIRECTIONS 4
#define cPATH_THROUGHPUTS 5
#define cPOTENTIAL_CONTRIBUTIONS 6
#define cSKYBOX 7
#define cMATERIALS 8
#define cTRIANGLES 9
#define cINVERSE_TRANSFORMS 10
#define cTEXTURE_RGBA32 11
#define cTEXTURE_RGBA128 12
#define cACCUMULATION_BUFFER 13
#define cAREALIGHT_BUFFER 14
#define cPOINTLIGHT_BUFFER 15
#define cSPOTLIGHT_BUFFER 16
#define cDIRECTIONALLIGHT_BUFFER 17
#define cBLUENOISE 18

// finalize bindings
#define fACCUMULATION_BUFFER 0
#define fUNIFORM_CONSTANTS 1
#define fOUTPUT 2

// Stage indices
#define STAGE_PRIMARY_RAY 0
#define STAGE_SECONDARY_RAY 1
#define STAGE_SHADOW_RAY 2
#define STAGE_SHADE 3
#define STAGE_FINALIZE 4

#define MAXPATHLENGTH 3
#define MAX_TRIANGLE_BUFFERS 65536
#define BLUENOISE 1

#endif // RENDERINGFW_VULKANRTX_SRC_BINDINGS_H

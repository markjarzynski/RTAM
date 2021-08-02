#include "Renderer.h"

#include <optix_function_table_definition.h>

namespace rtam {

    extern "C" char embedded_ptx_code[];

    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void *data;
    };

    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void *data;
    };

    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        int objectID;
    };

    Renderer::Renderer() {
        initOptix();
        createContext();
        createModule();
        /*
        createRaygenPrograms();
        createMissPrograms();
        createHitgroupPrograms();
        createPipline();
        buildSBT();
        launchParamsBuffer.alloc(sizeof(launchParams));
        */
    }

    void Renderer::initOptix() {
        std::cout << "Initializing Optix." << std::endl;

        cudaFree(0);
        int numDevices;
        cudaGetDeviceCount(&numDevices);

        if (numDevices == 0) {
            throw std::runtime_error("Error no CUDA capable devices found!");
        }

        std::cout << "Found " << numDevices << " CUDA devices." << std::endl;

        OPTIX_CHECK( optixInit() );
    }

    static void context_log_cb(unsigned int level, const char *tag, const char *message, void *) {
        fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
    }

    void Renderer::createContext() {
        const int deviceID = 0;
        CUDA_CHECK(cudaSetDevice(deviceID));
        CUDA_CHECK(cudaStreamCreate(&stream));

        cudaGetDeviceProperties(&deviceProps, deviceID);
        std::cout << "Running on device: " << deviceProps.name << std::endl;

        CUresult cudaResult = cuCtxGetCurrent(&cudaContext);
        if (cudaResult != CUDA_SUCCESS) {
            fprintf(stderr, "Error querying current context: %d\n", cudaResult);
        }

        OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
        OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
    }

    void Renderer::createModule() {

        moduleCompileOptions = {};

        moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT; // OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE; // OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

        pipelineCompileOptions = {};

        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.usesMotionBlur = 0;
        pipelineCompileOptions.numPayloadValues = 2;
        pipelineCompileOptions.numAttributeValues = 2;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

        pipelineLinkOptions.maxTraceDepth = 2;
        pipelineLinkOptions.debugLevel    = OPTIX_COMPILE_DEBUG_LEVEL_NONE; // OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO

        const std::string ptxCode = embedded_ptx_code;

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixModuleCreateFromPTX(optixContext, &moduleCompileOptions, &pipelineCompileOptions, ptxCode.c_str(), ptxCode.size(), log, &sizeof_log, &module));
        if (sizeof_log > 1) {
            fprintf(stderr, "%s\n", log);
        }
    }

    void Renderer::createRaygenPrograms() {

    }

    void Renderer::createMissPrograms() {

    }

    void Renderer::createHitgroupPrograms() {

    }

    void Renderer::createPipline() {

    }

    void Renderer::buildSBT() {

    }

    void Renderer::resize(const int2 &newSize) {
        // do something
    }

    void Renderer::downloadPixels(uint32_t pixels[]) {
        // do something
    }

    void Renderer::render() {
        // do something
    }


}

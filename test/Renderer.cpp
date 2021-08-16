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

        std::cout << "Creating OptiX context." << std::endl;
        createContext();

        std::cout << "Creating module." << std::endl;
        createModule();

        std::cout << "Creating Raygen programs." << std::endl;
        createRaygenPrograms();

        std::cout << "Creating Miss programs." << std::endl;
        createMissPrograms();

        std::cout << "Creating Hitgroup programs." << std::endl;
        createHitgroupPrograms();

        std::cout << "Creating Pipeline." << std::endl;
        createPipline();

        std::cout << "Building SBT." << std::endl;
        buildSBT();

        std::cout << "Allocating LaunchParamsBuffer." << std::endl;
        launchParamsBuffer.alloc(sizeof(launchParams));
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
        if (sizeof_log > 1) PRINT(log);
    }

    void Renderer::createRaygenPrograms() {
        raygenProgramGroups.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.raygen.module = module;
        pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &raygenProgramGroups[0]));
        if (sizeof_log > 1) PRINT(log);
    }

    void Renderer::createMissPrograms() {
        missProgramGroups.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.miss.module = module;
        pgDesc.miss.entryFunctionName = "__miss__radiance";

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &missProgramGroups[0]));
        if (sizeof_log > 1) PRINT(log);
    }

    void Renderer::createHitgroupPrograms() {
        hitgroupProgramGroups.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH = module;
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        pgDesc.hitgroup.moduleAH = module;
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &hitgroupProgramGroups[0]));
        if (sizeof_log > 1) PRINT(log);
    }

    void Renderer::createPipline() {

        std::vector<OptixProgramGroup> programGroups;

        for (OptixProgramGroup pg : raygenProgramGroups) {
            programGroups.push_back(pg);
        }

        for (OptixProgramGroup pg : missProgramGroups) {
            programGroups.push_back(pg);
        }

        for (OptixProgramGroup pg : hitgroupProgramGroups) {
            programGroups.push_back(pg);
        }

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(optixContext, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(), (int)programGroups.size(), log, &sizeof_log, &pipeline));
        if (sizeof_log > 1) PRINT(log);

        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 2048, 2048, 2048, 1));
        if (sizeof_log > 1) PRINT(log);
    }

    void Renderer::buildSBT() {

        std::vector<RaygenRecord> raygenRecords;
        for (OptixProgramGroup pg : raygenProgramGroups) {
            RaygenRecord raygen_record;
            OPTIX_CHECK(optixSbtRecordPackHeader(pg, &raygen_record));
            raygen_record.data = nullptr;
            raygenRecords.push_back(raygen_record);
        }
        raygenRecordsBuffer.alloc_and_upload(raygenRecords);
        sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

        std::vector<MissRecord> missRecords;
        for (OptixProgramGroup pg : missProgramGroups) {
            MissRecord miss_record;
            OPTIX_CHECK(optixSbtRecordPackHeader(pg, &miss_record));
            miss_record.data = nullptr;
            missRecords.push_back(miss_record);
        }
        missRecordsBuffer.alloc_and_upload(missRecords);
        sbt.missRecordBase = missRecordsBuffer.d_pointer();
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount = (int)missRecords.size();

        int numObjects = 1;
        std::vector<HitgroupRecord> hitgroupRecords;
        for (int i = 0; i < numObjects; i++) {
            int objectType = 0;
            HitgroupRecord hitgroup_record;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupProgramGroups[objectType], &hitgroup_record));
            hitgroup_record.objectID = i;
            hitgroupRecords.push_back(hitgroup_record);
        }
        hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
    }

    void Renderer::render() {
        std::cout << launchParams.frame.size.x << " " << launchParams.frame.size.y << std::endl;

        if (launchParams.frame.size.x == 0) return;

        launchParamsBuffer.upload(&launchParams,1);
        launchParams.frame.ID++;

        OPTIX_CHECK(optixLaunch(pipeline, stream, launchParamsBuffer.d_pointer(), launchParamsBuffer.sizeInBytes, &sbt, launchParams.frame.size.x, launchParams.frame.size.y, 1));
        CUDA_SYNC_CHECK();
    }

    void Renderer::resize(const int2 &newSize) {
        if (newSize.x == 0 | newSize.y == 0) return;

        colorBuffer.resize(newSize.x*newSize.y*sizeof(uint32_t));

        launchParams.frame.size = newSize;
        launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_ptr;
    }

    void Renderer::downloadPixels(uint32_t pixels[]) {
        colorBuffer.download(pixels, launchParams.frame.size.x * launchParams.frame.size.y);
    }

}

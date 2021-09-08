#include "Renderer.h"

#include <optix_function_table_definition.h>

#include <chrono>

#define PRINT_MS(PREVIOUS, CURRENT) \
    CURRENT = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); \
    std::cout << (CURRENT - PREVIOUS) / 1000 << "." << (CURRENT - PREVIOUS) % 1000 << "ms" << std::endl; \
    previous = current;

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

    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        TriangleMeshSBTData data;
    };

    Renderer::Renderer(const World *w) : world(w) {
        
         setCamera(world->camera);
        setBackground(world->background);


        auto previous = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        auto current = previous;

        initOptix();
        std::cout << "Initializing Optix...";
        PRINT_MS(previous,current);

        createContext();
        std::cout << "Creating OptiX context... ";
        PRINT_MS(previous,current);

        std::cout << "Creating module... ";
        createModule();
        PRINT_MS(previous,current);

        std::cout << "Creating Raygen programs... ";
        createRaygenPrograms();
        PRINT_MS(previous,current);

        std::cout << "Creating Miss programs... ";
        createMissPrograms();
        PRINT_MS(previous,current);

        std::cout << "Creating Hitgroup programs... ";
        createHitgroupPrograms();
        PRINT_MS(previous,current);

        std::cout << "Building Accel... ";
        launchParams.traversable = buildAccel();
        PRINT_MS(previous,current);

        std::cout << "Creating Pipeline... ";
        createPipline();
        PRINT_MS(previous,current);

        std::cout << "Building SBT... ";
        buildSBT();
        PRINT_MS(previous,current);

        std::cout << "Allocating LaunchParamsBuffer... ";
        launchParamsBuffer.alloc(sizeof(launchParams));
        PRINT_MS(previous,current);
    }

    void Renderer::initOptix() {

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
        //if (sizeof_log > 1) PRINT(log);
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
        //if (sizeof_log > 1) PRINT(log);
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
        //if (sizeof_log > 1) PRINT(log);
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
        //if (sizeof_log > 1) PRINT(log);
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
        //if (sizeof_log > 1) PRINT(log);

        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 2048, 2048, 2048, 1));
        //if (sizeof_log > 1) PRINT(log);
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

        std::vector<HitgroupRecord> hitgroupRecords;
        for (int i = 0; i < world->triangles.size(); i++) {
            Triangle t = world->triangles[i];
            HitgroupRecord hitgroup_record;

            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupProgramGroups[0], &hitgroup_record));
            hitgroup_record.data.color = t.diffuse;
            hitgroup_record.data.vertex = (float3*)vertexBuffer[i].d_pointer();
            hitgroup_record.data.normal = (float3*)normalBuffer[i].d_pointer();
            hitgroup_record.data.index = (int3*)indexBuffer[i].d_pointer();
            hitgroupRecords.push_back(hitgroup_record);
        }
        hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
    }

    OptixTraversableHandle Renderer::buildAccel() {
        vertexBuffer.resize(world->triangles.size());
        normalBuffer.resize(world->triangles.size());
        indexBuffer.resize(world->triangles.size());

        OptixTraversableHandle asHandle { 0 };

        std::vector<OptixBuildInput> triangleInput(world->triangles.size());
        std::vector<CUdeviceptr> d_vertices(world->triangles.size());
        std::vector<CUdeviceptr> d_indices(world->triangles.size());
        std::vector<uint32_t> triangleInputFlags(world->triangles.size());

        for (int i = 0; i < world->triangles.size(); i++) {
            Triangle t = world->triangles[i];
            vertexBuffer[i].alloc_and_upload(t.v);
            normalBuffer[i].alloc_and_upload(t.n);
            indexBuffer[i].alloc_and_upload(t.i);

            triangleInput[i] = {};
            triangleInput[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            d_vertices[i] = vertexBuffer[i].d_pointer();
            d_indices[i] = indexBuffer[i].d_pointer();

            triangleInput[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangleInput[i].triangleArray.vertexStrideInBytes = sizeof(float3);
            triangleInput[i].triangleArray.numVertices = t.v.size();
            triangleInput[i].triangleArray.vertexBuffers = &d_vertices[i];

            triangleInput[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangleInput[i].triangleArray.indexStrideInBytes = sizeof(int3);
            triangleInput[i].triangleArray.numIndexTriplets = t.i.size();
            triangleInput[i].triangleArray.indexBuffer = d_indices[i];

            triangleInputFlags[i] = 0;

            triangleInput[i].triangleArray.flags = &triangleInputFlags[i];
            triangleInput[i].triangleArray.numSbtRecords = 1;
            triangleInput[i].triangleArray.sbtIndexOffsetBuffer = 0;
            triangleInput[i].triangleArray.sbtIndexOffsetSizeInBytes = 0;
            triangleInput[i].triangleArray.sbtIndexOffsetStrideInBytes = 0;
        }

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.motionOptions.numKeys = 1;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes blasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions, triangleInput.data(), world->triangles.size(), &blasBufferSizes));

        CUDABuffer compactedSizeBuffer;
        compactedSizeBuffer.alloc(sizeof(uint64_t));

        OptixAccelEmitDesc emitDesc;
        emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = compactedSizeBuffer.d_pointer();

        CUDABuffer tempBuffer;
        tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

        CUDABuffer outputBuffer;
        outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

        OPTIX_CHECK(optixAccelBuild(optixContext, 0, &accelOptions, triangleInput.data(), world->triangles.size(), tempBuffer.d_pointer(), tempBuffer.sizeInBytes, outputBuffer.d_pointer(), outputBuffer.sizeInBytes, &asHandle, &emitDesc, 1));
        CUDA_SYNC_CHECK();

        uint64_t compactedSize;
        compactedSizeBuffer.download(&compactedSize,1);

        asBuffer.alloc(compactedSize);
        OPTIX_CHECK(optixAccelCompact(optixContext, 0, asHandle, asBuffer.d_pointer(), asBuffer.sizeInBytes, &asHandle));
        CUDA_SYNC_CHECK();

        outputBuffer.free();
        tempBuffer.free();
        compactedSizeBuffer.free();

        return asHandle;
    }

    void Renderer::render() {

        if (launchParams.frame.size.x == 0) return;

        auto previous = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        auto current = previous;

        std::cout << "Rendering... ";

        launchParamsBuffer.upload(&launchParams,1);
        launchParams.frame.ID++;

        OPTIX_CHECK(optixLaunch(pipeline, stream, launchParamsBuffer.d_pointer(), launchParamsBuffer.sizeInBytes, &sbt, launchParams.frame.size.x, launchParams.frame.size.y, 1));
        CUDA_SYNC_CHECK();

        PRINT_MS(previous, current);
    }

    void Renderer::setCamera(Camera camera) {
        lastCamera = camera;
        camera.setAspectRatio(static_cast<float>(launchParams.frame.size.x) / static_cast<float>(launchParams.frame.size.y));
        launchParams.camera.eye = camera.eyep;
        camera.UVWFrame(launchParams.camera.U, launchParams.camera.V, launchParams.camera.W);
    }

    void Renderer::setBackground(float3 background) {
        launchParams.background = background;
    }

    void Renderer::resize(const int2 &newSize) {
        if (newSize.x == 0 | newSize.y == 0) return;

        colorBuffer.resize(newSize.x*newSize.y*sizeof(uint32_t));

        launchParams.frame.size = newSize;
        launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_ptr;

        setCamera(lastCamera);
    }

    void Renderer::downloadPixels(uint32_t pixels[]) {
        colorBuffer.download(pixels, launchParams.frame.size.x * launchParams.frame.size.y);
    }

}

#pragma once

#include "CUDABuffer.h"
#include "LaunchParams.h"

#include "sutil/vec_math.h"

#include <optix.h>
#include <optix_stubs.h>

namespace rtam {

    class Renderer {
    public:
        Renderer();

    public:
        void render();

        void resize(const int2 &newSize);

        void downloadPixels(uint32_t pixels[]);

    protected:
        void initOptix();
        void createContext();
        void createModule();
        void createRaygenPrograms();
        void createMissPrograms();
        void createHitgroupPrograms();
        void createPipline();
        void buildSBT();

    protected:
        CUcontext cudaContext;
        CUstream stream;
        cudaDeviceProp deviceProps;

        OptixDeviceContext optixContext;

        OptixPipeline pipeline;
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        OptixPipelineLinkOptions pipelineLinkOptions = {};

        OptixModule module;
        OptixModuleCompileOptions moduleCompileOptions;

        std::vector<OptixProgramGroup> raygenProgramGroups;
        CUDABuffer raygenRecordsBuffer;
        std::vector<OptixProgramGroup> missProgramGroups;
        CUDABuffer missRecordsBuffer;
        std::vector<OptixProgramGroup> hitProgramGroups;
        CUDABuffer hitRecordsBuffer;
        OptixShaderBindingTable sbt = {};

        LaunchParams launchParams;
        CUDABuffer launchParamsBuffer;

        CUDABuffer colorBuffer;
    };

}

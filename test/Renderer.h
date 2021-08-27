#pragma once

#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Camera.h"
#include "World.h"

namespace rtam {

    class Renderer {
    public:
        Renderer(const World *world);

    public:
        void render();

        void resize(const int2 &newSize);

        void downloadPixels(uint32_t pixels[]);

        void setCamera(Camera camera);

        void setBackground(float3 background);

    protected:
        void initOptix();
        void createContext();
        void createModule();
        void createRaygenPrograms();
        void createMissPrograms();
        void createHitgroupPrograms();
        void createPipline();
        void buildSBT();
        OptixTraversableHandle buildAccel();

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
        std::vector<OptixProgramGroup> hitgroupProgramGroups;
        CUDABuffer hitgroupRecordsBuffer;
        OptixShaderBindingTable sbt = {};

        LaunchParams launchParams;
        CUDABuffer launchParamsBuffer;

        CUDABuffer colorBuffer;

        Camera lastCamera;

        const World *world;

        std::vector<CUDABuffer> vertexBuffer;
        std::vector<CUDABuffer> normalBuffer;
        std::vector<CUDABuffer> indexBuffer;
        CUDABuffer asBuffer;
    };

}

#include "Renderer.h"

#include <optix_function_table_definition.h>

#include <iostream>

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
        /*
        createModule();
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
        //CUDA_CHECK(SetDevice(deviceID));
        //CUDA_CHECK(StreamCreate(&stream));

        cudaGetDeviceProperties(&deviceProps, deviceID);
        std::cout << "Running on device: " << deviceProps.name << std::endl;

        CUresult cudaResult = cuCtxGetCurrent(&cudaContext);
        if (cudaResult != CUDA_SUCCESS) {
            //throw std::runtime_error("Error querying current context: %d\n", cudaResult);
        }

        OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
        OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
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

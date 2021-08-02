#include "Renderer.h"

#include <optix_function_table_definition.h>

#include <iostream>

namespace rtam {

    Renderer::Renderer() {
        initOptix();

    }

    void Renderer::initOptix() {
        std::cout << "initializing optix." << std::endl;

        cudaFree(0);
        int numDevices;
        cudaGetDeviceCount(&numDevices);

        if (numDevices == 0) {
            throw std::runtime_error("no CUDA capable devices found!");
        }

        std::cout << "found " << numDevices << " CUDA devices" << std::endl;

        OPTIX_CHECK( optixInit() );
    }

    void Renderer::render() {
        // do something
    }
}

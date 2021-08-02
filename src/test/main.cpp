#include "optix7.h"

#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"

namespace rtam {

    void initOptix() {
        cudaFree(0);
        int numDevices;
        cudaGetDeviceCount(&numDevices);

        if (numDevices == 0) {
            throw std::runtime_error("no CUDA capable devices found!");
        }

        std::cout << "found " << numDevices << " CUDA devices" << std::endl;

        OPTIX_CHECK( optixInit() );
    }

    extern "C" int main(int ac, char **av) {
        try {
            std::cout << "initializing optix." << std::endl;

            initOptix();

            std::cout << "successfully initialized optix." << std::endl;
            std::cout << "done. clean exit." << std::endl;
        } catch (std::runtime_error& e) {
            std::cout << "FATAL ERROR: " << e.what() << std::endl;
            exit(1);
        }

        return 0;
    }

};

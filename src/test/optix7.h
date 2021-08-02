#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>


#define OPTIX_CHECK(call) {                                                     \
    OptixResult res = call;                                                     \
    if( res != OPTIX_SUCCESS ) {                                                \
        fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
        exit( 2 );                                                              \
    }                                                                           \
}

#define CUDA_CHECK(call) {                                                      \
    cudaError_t cuda_error cuda##call;                                          \
    if (cuda_error != cudaSuccess) {                                            \
        std::stringstream txt;                                                  \
        txt << "CUDA Error " << cudaGetErrorName(cuda_error) << " (" << cudaGetErrorString(cuda_error) << ")"; \
        throw std::runtime_error(txt.str());                                    \
    }                                                                           \
}

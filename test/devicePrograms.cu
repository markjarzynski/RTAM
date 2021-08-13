#include <optix_device.h>

#include "LaunchParams.h"

namespace rtam {

    extern "C" __constant__ LaunchParams optixLaunchParams;

    extern "C" __global__ void __closesthit__radiance() {}

    extern "C" __global__ void __anyhit__radiance() {}

    extern "C" __global__ void __miss__radiance() {}

    extern "C" __global__ void __raygen__renderFrame() {
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;

        const uint32_t rgba = 0xff0000ff;

        const uint32_t index = ix + iy * optixLaunchParams.frame.size.x;
        optixLaunchParams.frame.colorBuffer[index] = rgba;
    }
}

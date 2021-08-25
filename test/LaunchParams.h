#pragma once

#include <sutil/vec_math.h>

#include <cstdint>

namespace rtam {

    struct TriangleMeshSBTData {
        float3 color;
        float3 *vertex;
        float3 *index;
        float3 *normal;
    };

    struct LaunchParams {
        struct {
            int32_t ID { 0 };
            uint32_t *colorBuffer;
            int2 size;
        } frame;

        struct {
            float3 eye;
            float3 U;
            float3 V;
            float3 W;
        } camera;

        OptixTraversableHandle traversable;
    };

}

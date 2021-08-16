#pragma once

#include <sutil/vec_math.h>
#include <sutil/Camera.h>

#include <cstdint>

namespace rtam {

    struct LaunchParams {
        struct {
            int32_t ID { 0 };
            uint32_t *colorBuffer;
            int2 size;
        } frame;

        //Camera camera;

        OptixTraversableHandle traversable;
    };

}

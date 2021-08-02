#pragma once

#include "sutil/vec_math.h"

#include <cstdint>

namespace rtam {

    struct LaunchParams {
        int frameID { 0 };
        uint32_t *colorBuffer;
        int2 fbSize;
    };

}

#pragma once

#include "CUDABuffer.h"

#include <sutil/vec_math.h>
#include <sutil/Camera.h>

#include <string>
#include <fstream>
#include <iostream>

namespace rtam {
    class World {
    public:
        std::string outfile = "test.png";
        float3 background;
        float3 eyep;
        float3 lookp;
        float3 up;
        float2 fov;
        int2 screen;

        //sutil::Camera camera;

    public:
        World (std::string filename);

    private:
        void read_rayshade(std::string filename);
    };
}

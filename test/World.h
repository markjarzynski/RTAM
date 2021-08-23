#pragma once

#include "CUDABuffer.h"
#include "Camera.h"

#include <sutil/vec_math.h>

#include <string>
#include <fstream>
#include <iostream>

namespace rtam {
    class World {
    public:
        std::string outfile = "test.png";
        float3 background;
        Camera camera;
        int2 screen;

        //sutil::Camera camera;

    public:
        World (std::string filename);

    private:
        void read_rayshade(std::string filename);
    };
}

#pragma once

#include "optix7.h"

#include "LaunchParams.h"

namespace rtam {

    class Renderer {
    public:
        Renderer();

    public:
        void render();

    protected:
        void initOptix();

    };

}

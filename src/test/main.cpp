#include "Renderer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"

#include <iostream>

namespace rtam {

    extern "C" int main(int ac, char **av) {
        try {
            Renderer renderer;
            renderer.render();

            std::cout << "successfully initialized optix." << std::endl;
            std::cout << "done. clean exit." << std::endl;
        } catch (std::runtime_error& e) {
            std::cout << "FATAL ERROR: " << e.what() << std::endl;
            exit(1);
        }

        return 0;
    }

};

#include "Renderer.h"
#include "World.h"

#include "sutil/vec_math.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"

#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace rtam {

    extern "C" int main(int ac, char **av) {
        try {
            std::cout << fs::current_path() << std::endl;

            World world = World(
                #ifdef _WIN32 
                "../../data/tetra.ray"
                #else
                "../data/tetra.ray"
                #endif
            );

            Renderer renderer;

            int2 fbSize = make_int2(1024,1024);

            renderer.render();

            std::vector<uint32_t> pixels (fbSize.x * fbSize.y);
            renderer.downloadPixels(pixels.data());

            const std::string filename = "test.png";
            stbi_write_png(filename.c_str(), fbSize.x, fbSize.y, 4, pixels.data(), fbSize.x * sizeof(uint32_t));

            std::cout << "Image rendered to " << filename << std::endl;
        } catch (std::runtime_error& e) {
            std::cout << "FATAL ERROR: " << e.what() << std::endl;
            exit(1);
        }

        return 0;
    }

};

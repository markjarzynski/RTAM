#include "Renderer.h"
#include "World.h"

#include "sutil/vec_math.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"

#include <iostream>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

namespace rtam {

    extern "C" int main(int ac, char **av) {

        // Get the start time in milliseconds so we can figure out how long it took this program to execute
        auto start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        try {

            std::cout << fs::current_path() << std::endl;

            World world = World(
                #ifdef _WIN32
                "../../data/tetra-9.ray"
                #else
                "../data/tetra-9.ray"
                #endif
            );

            Renderer renderer = Renderer(&world);

            int2 fsize = world.screen;
            renderer.resize(fsize);
            renderer.render();

            std::vector<uint32_t> pixels (fsize.x * fsize.y);
            renderer.downloadPixels(pixels.data());

            const std::string filename = "test.png";
            stbi_write_png(filename.c_str(), fsize.x, fsize.y, 4, pixels.data(), fsize.x * sizeof(uint32_t));

            std::cout << "Image rendered to " << filename << std::endl;
        } catch (std::runtime_error& e) {
            std::cout << "FATAL ERROR: " << e.what() << std::endl;
            exit(1);
        }

        // Get the end time and convert to "seconds.milliseconds"
        auto end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        auto milliseconds = end_time - start_time;
        auto seconds = milliseconds / 1000;
        milliseconds %= 1000;
        // print out seconds.milliseconds seconds
        std::cout << "Total time: " << seconds << "." << milliseconds << " seconds" << std::endl;

        return 0;
    }

};

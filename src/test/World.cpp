#include "World.h"

namespace rtam {

    World::World (std::string filename) {
        read_rayshade(filename);
    }

    // read a rayshade (.ray) file
    void World::read_rayshade (std::string filename) {

        PRINT(filename);

        std::ifstream stream (filename, std::ifstream::in);
        std::string key;

        std::cout << stream.is_open() << std::endl;

        while (stream >> key) {

            std::cout << key << std::endl;

            if (key == "outfile") {
                stream >> outfile;
            } else if (key == "background") {
                float r, g, b;
                stream >> r >> g >> b;
                background = make_float3(r,g,b);

                std::cout << background.x << background.y << background.z << std::endl;
            }

        }

        stream.close();

        std::cout << filename << std::endl;

    }
};

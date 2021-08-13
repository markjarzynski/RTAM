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

            if (key == "outfile") {
                stream >> outfile;
            } else if (key == "background") {
                float r, g, b;
                stream >> r >> g >> b;
                background = make_float3(r,g,b);
            } else if (key == "eyep") {
                float x, y, z;
                stream >> x >> y >> z;
                eyep = make_float3(x,y,z);
                camera.setEye(eyep);
            } else if (key == "lookp") {
                float x, y, z;
                stream >> x >> y >> z;
                lookp = make_float3(x,y,z);
                camera.setLookat(lookp);
            } else if (key == "up") {
                float x, y, z;
                stream >> x >> y >> z;
                up = make_float3(x,y,z);
                camera.setUp(up);
            } else if (key == "fov") {
                float hfov, vfov;
                stream >> hfov >> vfov;
                fov = make_float2(hfov, vfov);
                camera.setFovY(vfov);
                camera.setAspectRatio(hfov/vfov);
            } else if (key == "screen") {
                float w, h;
                stream >> w >> h;
                screen = make_int2(w, h);
            }

        }

        stream.close();

        std::cout << filename << std::endl;

    }
};

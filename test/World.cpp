#include "World.h"

#include <filesystem>

namespace rtam {

    World::World (std::string filename) {
        read_rayshade(filename);
    }

    // read a rayshade (.ray) file
    void World::read_rayshade (std::string filename) {

        std::cout << "Reading file: " << std::filesystem::absolute(filename) << std::endl;

        std::ifstream stream (filename, std::ifstream::in);
        std::string key;

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
                //camera.setEye( {x, y, z} );
            } else if (key == "lookp") {
                float x, y, z;
                stream >> x >> y >> z;
                lookp = make_float3(x,y,z);
                //camera.setLookat(lookp);
            } else if (key == "up") {
                float x, y, z;
                stream >> x >> y >> z;
                up = make_float3(x,y,z);
                //camera.setUp(up);
            } else if (key == "fov") {
                float hfov, vfov;
                stream >> hfov >> vfov;
                fov = make_float2(hfov, vfov);
                //camera.setFovY(hfov);
            } else if (key == "screen") {
                int w, h;
                stream >> w >> h;
                screen = make_int2(w, h);
                //camera.setAspectRatio((float)w / (float)h);
            }

        }

        stream.close();
    }

};

#include "World.h"

#include <filesystem>

namespace rtam {

    World::World (std::string filename) {
        //camera = Camera();
        read_rayshade(filename);
    }

    // read a rayshade (.ray) file
    void World::read_rayshade (std::string filename) {

        std::cout << "Reading file: " << std::filesystem::absolute(filename) << std::endl;

        std::ifstream stream (filename, std::ifstream::in);
        std::string key;

        float3 diffuse = make_float3(1,1,1);

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
                camera.eyep = make_float3(x,y,z);
            } else if (key == "lookp") {
                float x, y, z;
                stream >> x >> y >> z;
                camera.lookp = make_float3(x,y,z);
            } else if (key == "up") {
                float x, y, z;
                stream >> x >> y >> z;
                camera.up = make_float3(x,y,z);
            } else if (key == "fov") {
                float hfov, vfov;
                stream >> hfov >> vfov;
                camera.fov = make_float2(hfov, vfov);
            } else if (key == "screen") {
                int w, h;
                stream >> w >> h;
                screen = make_int2(w, h);
                //camera.setAspectRatio( static_cast<float>( w ) / static_cast<float>( h ) );
            } else if (key == "polygon") {
                // our current test file tetra.ray is made up of polygons with only 3 verticies, so using triangles right now

                std::string mat;
                float x, y, z;
                float3 a, b, c;
                Triangle tri;

                stream >> mat;
                stream >> x >> y >> z;
                a = make_float3(x, y, z);
                stream >> x >> y >> z;
                b = make_float3(x, y, z);
                stream >> x >> y >> z;
                c = make_float3(x, y, z);

                tri = Triangle(a,b,c,diffuse);

                triangles.push_back(tri);
            } else if (key == "diffuse") {
                float x, y, z;
                stream >> x >> y >> z;
                diffuse = make_float3(x,y,z);
            }

        }

        stream.close();

        // Print out what we read in for sanity checks.
        std::cout << "background " << background.x << " " << background.y << " " << background.z << std::endl;
        std::cout << "eyep " << camera.eyep.x << " " << camera.eyep.y << " " << camera.eyep.z << std::endl;
        std::cout << "lookp " << camera.lookp.x << " " << camera.lookp.y << " " << camera.lookp.z << std::endl;
        std::cout << "up " << camera.up.x << " " << camera.up.y << " " << camera.up.z << std::endl;
        std::cout << "fov " << camera.fov.x << " " << camera.fov.y << std::endl;
        std::cout << "screen " << screen.x << " " << screen.y << std::endl;

        std::cout << "diffuse " << diffuse.x << " " << diffuse.y << " " << diffuse.z << std::endl;

        /*
        for (auto tri : triangles) {
            std::cout << "polygon " << tri.v[0].x << " " << tri.v[0].y << " " << tri.v[0].z
                             << " " << tri.v[1].x << " " << tri.v[1].y << " " << tri.v[1].z
                             << " " << tri.v[2].x << " " << tri.v[2].y << " " << tri.v[2].z << std::endl;
        }
        */

        std::cout << "faces: " << triangles.size() << std::endl;
        std::cout << "vertices: " << triangles.size() * 3 << std::endl;
        std::cout << "edges: " << triangles.size() * 3 << std::endl;



    }

};

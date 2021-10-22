#pragma once

#include <sutil/vec_math.h>

#include <iostream>
#include <fstream>

namespace rtam {

    class Camera {
    public:
        float3 eyep;
        float3 lookp;
        float3 up;
        float2 fov;
        float aspect_ratio;

    public:
        Camera() {
            eyep = make_float3(0, -10, 0);
            lookp = make_float3(0, 0, 0);
            up = make_float3(0, 0, 1);
            fov = make_float2(45,45);
        }

        Camera(float3 eyep, float3 lookp, float3 up) : eyep(eyep), lookp(lookp), up(up) {
            fov = make_float2(45,45);
        }

        Camera(float3 eyep, float3 lookp, float3 up, float2 fov) : eyep(eyep), lookp(lookp), up(up), fov(fov) {}

    public:
        void setAspectRatio(float ratio) {
            aspect_ratio = ratio;
        }

        void UVWFrame(float3& U, float3& V, float3& W) const
        {
            /*
            W = lookp - eyep;
            U = normalize(cross(W, up));
            V = normalize(cross(U, W));

            float len = length(W) * tanf(0.5f * fov.y * M_PIf / 180.0f);
            V *= len;
            U *= len * aspect_ratio;
            */

            W = normalize(lookp - eyep);
            U = normalize(cross(W, up));
            V = normalize(cross(W, U));
        }

    };

    /*
    std::ostream& operator << (std::ostream& out, const Camera& camera) {
        out << "eyep " << camera.eyep.x;// << " " << camera.eyep.y << " " camera.eyep.z << std::endl;
        out << "lookp " << camera.lookp.x << " " << camera.lookp.y << " " camera.lookp.z << std::endl;
        out << "up " << camera.up.x << " " << camera.up.y << " " camera.up.z << std::endl;
        out << "fov " << camera.fov.x << " " << camera.fov.y << std::endl;

        return out;
    }
    */

}

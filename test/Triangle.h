#pragma once

#include <sutil/vec_math.h>

namespace rtam {

    class Triangle {
    public:
        float3 a, b, c;
        float3 n0, n1, n2;

    public:
        Triangle() {
            a = make_float3(1,0,0);
            b = make_float3(0,1,0);
            c = make_float3(0,0,1);
        }

        Triangle(float3 a, float3 b, float3 c) : a(a), b(b), c(c) {
            setNormal();
        }

    public:
        void setNormal() {
            n0 = n1 = n2 = normalize(cross(b - a, c - b));
        }

    };
}

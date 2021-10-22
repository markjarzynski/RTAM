#pragma once

#include <sutil/vec_math.h>

#include <vector>

namespace rtam {

    class Triangle {
    public:
        std::vector<float3> v;
        std::vector<float3> n;
        std::vector<unsigned int> i;
        float3 diffuse;

    public:
        Triangle() {
            v.push_back(make_float3(1,0,0)); v.push_back(make_float3(0,1,0)); v.push_back(make_float3(0,0,1));
            setNormal();
            setIndices();
            setDiffuse();
        }

        Triangle(float3 v0, float3 v1, float3 v2) {
            v.push_back(v0); v.push_back(v1); v.push_back(v2);
            setNormal();
            setIndices();
            setDiffuse();
        }

        Triangle(float3 v0, float3 v1, float3 v2, float3 diffuse) : diffuse(diffuse) {
            v.push_back(v0); v.push_back(v1); v.push_back(v2);
            setNormal();
            setIndices();
        }

        Triangle(float3 v0, float3 v1, float3 v2, float3 n0, float3 n1, float3 n2) {
            v.push_back(v0); v.push_back(v1); v.push_back(v2);
            n.push_back(n0); n.push_back(n1); n.push_back(n2);
            setIndices();
            setDiffuse();
        }

        Triangle(float3 v0, float3 v1, float3 v2, float3 n0, float3 n1, float3 n2, float3 diffuse) : diffuse(diffuse) {
            v.push_back(v0); v.push_back(v1); v.push_back(v2);
            n.push_back(n0); n.push_back(n1); n.push_back(n2);
            setIndices();
        }

    private:
        void setNormal() {
            float3 normal = normalize(cross(v[1] - v[0], v[2] - v[1]));
            n.push_back(normal); n.push_back(normal); n.push_back(normal);
        }

        void setIndices() {
            i.push_back(0); i.push_back(1); i.push_back(2);
        }

        void setDiffuse() {
            diffuse = make_float3(1,1,1);
        }

    };
}

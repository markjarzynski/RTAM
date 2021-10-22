#include <sutil/Preprocessor.h>
#include <sutil/vec_math.h>

namespace rtam {
    SUTIL_INLINE SUTIL_HOSTDEVICE uint3 pcg3d (uint3 v) {

        v = v * make_uint3(1664525u, 1664525u, 1664525u) + make_uint3(1013904223u, 1013904223u, 1013904223u);

        v.x += v.y*v.z;
        v.y += v.z*v.x;
        v.z += v.x*v.y;

        v.x = v.x ^ v.x >> 16u;
        v.y = v.y ^ v.y >> 16u;
        v.z = v.z ^ v.z >> 16u;

        v.x += v.y*v.z;
        v.y += v.z*v.x;
        v.z += v.x*v.y;

        return v;
    }
}

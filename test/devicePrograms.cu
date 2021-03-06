#include <optix_device.h>

#include "LaunchParams.h"
#include "Random.h"

namespace rtam {

    extern "C" __constant__ LaunchParams optixLaunchParams;

    enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };

    static __forceinline__ __device__
    void *unpackPointer( uint32_t i0, uint32_t i1 )
    {
      const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
      void*           ptr = reinterpret_cast<void*>( uptr );
      return ptr;
    }

    static __forceinline__ __device__
    void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
    {
      const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
      i0 = uptr >> 32;
      i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T *getPRD()
    {
      const uint32_t u0 = optixGetPayload_0();
      const uint32_t u1 = optixGetPayload_1();
      return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
    }

    extern "C" __global__ void __closesthit__radiance() {
        const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

        const int i = optixGetPrimitiveIndex();
        const int3 index = sbtData.index[i];

        const float3 rayd = optixGetWorldRayDirection();
        const float3 normal = sbtData.normal[index.x];

        const float3 v = (sbtData.vertex[index.x] + sbtData.vertex[index.y] + sbtData.vertex[index.z]) / 3.0f;

        /*
        float3 &prd = *(float3*)getPRD<float3>();
        prd = (0.2f + 0.8f * fabsf(dot(rayd,normal))) * sbtData.color;
        */

        /*
        float4 &prd = *(float4*)getPRD<float4>();
        prd = make_float4((0.2f + 0.8f * fabsf(dot(rayd,normal))) * sbtData.color, 1.0f);
        */

        const float4 c = make_float4((0.2f + 0.8f * fabsf(dot(rayd,normal))) * sbtData.color, 1.0f);

        floatx &prd = *(floatx*)getPRD<floatx>();
        for (int i = 0; i < FLOATX; i+=4) {
            prd[i] = c.x;
            prd[i+1] = c.y;
            prd[i+2] = c.z;
            prd[i+3] = c.w;
        }
    }

    extern "C" __global__ void __anyhit__radiance() { }

    extern "C" __global__ void __miss__radiance() {
        /*
        float3 &prd = *(float3*)getPRD<float3>();
        prd = make_float3(0.f,0.f,0.f);
        prd = optixLaunchParams.background;
        */

        /*
        float4 &prd = *(float4*)getPRD<float4>();
        prd = make_float4(0.f,0.f,0.f,0.f);
        prd = make_float4(optixLaunchParams.background, 0.f);
        */

        const float4 c = make_float4(optixLaunchParams.background, 0.f);

        floatx &prd = *(floatx*)getPRD<floatx>();
        for (int i = 0; i < FLOATX; i+=4) {
            prd[i] = c.x;
            prd[i+1] = c.y;
            prd[i+2] = c.z;
            prd[i+3] = c.w;
        }
    }

    extern "C" __global__ void __raygen__renderFrame() {
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;

        const auto &camera = optixLaunchParams.camera;

        //float3 pixelColorPRD = make_float3(0.f,0.f,0.f);
        //float4 pixelColorPRD = make_float4(0.f,0.f,0.f,1.f);
        floatx pixelColorPRD;

        uint32_t u0, u1;
        packPointer( &pixelColorPRD, u0, u1 );

        const float2 screen = make_float2(ix + 0.5f, iy + 0.5f) / make_float2(optixLaunchParams.frame.size);

        const float3 rayd = normalize(camera.U * (screen.x - 0.5f) + camera.V * (screen.y - 0.5f) + camera.W);
        const float3 rayo = camera.eye;

        //atomicAdd(optixLaunchParams.frame.ray_count, 1);
        //optixLaunchParams.frame.ray_count++;

        optixTrace(optixLaunchParams.traversable, rayo, rayd, 0.f, 1e20f, 0.0f, OptixVisibilityMask( 255 ), OPTIX_RAY_FLAG_DISABLE_ANYHIT, SURFACE_RAY_TYPE, RAY_TYPE_COUNT, SURFACE_RAY_TYPE, u0, u1);

        /*
        const int r = int(255.99f*pixelColorPRD.x);
        const int g = int(255.99f*pixelColorPRD.y);
        const int b = int(255.99f*pixelColorPRD.z);
        const int a = int(255.99f*pixelColorPRD.w);
        */

        float c[4] = {0.f, 0.f, 0.f, 0.f};

        for (int i = 0; i < FLOATX; i++) {
            c[i%4] += pixelColorPRD[i] / (FLOATX / 4.f);
        }

        const int r = int(255.99f*c[0]);
        const int g = int(255.99f*c[1]);
        const int b = int(255.99f*c[2]);
        const int a = int(255.99f*c[3]);


        //const uint32_t rgba = 0xff000000 | r | (g<<8u) | (b<<16u);
        const uint32_t rgba = r | (g<<8u) | (b<<16u) | (a<<24u);

        const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
        optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
        
    }
}

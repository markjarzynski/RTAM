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

    static __forceinline__ __device__ void trace(OptixTraversableHandle handle, float3 rayo, float3 rayd, float tmin, float tmax, float4* prd)
    {
      uint32_t u0, u1;
      packPointer( prd, u0, u1 );
      optixTrace(handle, rayo, rayd, tmin, tmax, 0.0f, OptixVisibilityMask( 255 ), OPTIX_RAY_FLAG_DISABLE_ANYHIT, SURFACE_RAY_TYPE, RAY_TYPE_COUNT, SURFACE_RAY_TYPE, u0, u1);
    }

    extern "C" __global__ void __closesthit__radiance() {
        const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

        const int i = optixGetPrimitiveIndex();
        const int3 index = sbtData.index[i];

        const float3 rayd = optixGetWorldRayDirection();
        const float3 normal = sbtData.normal[index.x];

        const float3 v = (sbtData.vertex[index.x] + sbtData.vertex[index.y] + sbtData.vertex[index.z]) / 3.0f;

        float4 &prd = *(float4*)getPRD<float4>();
        prd = make_float4((0.2f + 0.8f * fabsf(dot(rayd,normal))) * sbtData.color, 1.0f);

        const float3 rayo = optixGetWorldRayOrigin() + optixGetRayTmax()*rayd;

        float4 prd2 = make_float4(0.f,0.f,0.f,1.f);
        trace(optixLaunchParams.traversable, rayo, rayd, 0.f, 1e20f, &prd2);

        prd += prd2;
    }

    extern "C" __global__ void __anyhit__radiance() { }

    extern "C" __global__ void __miss__radiance() {
        float4 &prd = *(float4*)getPRD<float4>();
        prd = make_float4(0.f,0.f,0.f,0.f);
        prd = make_float4(optixLaunchParams.background, 0.f);
    }

    extern "C" __global__ void __raygen__renderFrame() {
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;

        const auto &camera = optixLaunchParams.camera;

        float4 prd = make_float4(0.f,0.f,0.f,1.f);

        const float2 screen = make_float2(ix + 0.5f, iy + 0.5f) / make_float2(optixLaunchParams.frame.size);

        const float3 rayd = normalize(camera.U * (screen.x - 0.5f) + camera.V * (screen.y - 0.5f) + camera.W);
        const float3 rayo = camera.eye;

        //atomicAdd(optixLaunchParams.frame.ray_count, 1);
        //optixLaunchParams.frame.ray_count++;

        trace(optixLaunchParams.traversable, rayo, rayd, 0.f, 1e20f, &prd);

        const int r = int(255.99f*prd.x);
        const int g = int(255.99f*prd.y);
        const int b = int(255.99f*prd.z);
        const int a = int(255.99f*prd.w);
        
        const uint32_t rgba = 0xff000000 | r | (g<<8u) | (b<<16u);

        const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
        optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
        
    }
}

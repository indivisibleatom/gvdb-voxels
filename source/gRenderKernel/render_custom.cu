
#include <stdio.h>
#include "cuda_math.cuh"

typedef unsigned char   uchar;
typedef unsigned int    uint;
typedef unsigned short    ushort;
typedef unsigned long   ulong;
typedef unsigned long long  uint64;

//-------------------------------- GVDB Data Structure
#define CUDA_PATHWAY
#include "cuda_gvdb_scene.cuh"    // GVDB Scene
#include "cuda_gvdb_nodes.cuh"    // GVDB Node structure
#include "cuda_gvdb_geom.cuh"   // GVDB Geom helpers
#include "cuda_gvdb_dda.cuh"    // GVDB DDA
#include "cuda_gvdb_raycast.cuh"  // GVDB Raycasting
#include <curand.h>
#include <curand_kernel.h>
//--------------------------------


inline __host__ __device__ float3 reflect3 (float3 i, float3 n)
{
  return i - 2.0f * n * dot(n,i);
}

// Custom raycast kernel
extern "C" __global__ void raycast_kernel ( uchar4* outBuf )
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  curandState_t state;
  curand_init(0,0,0,&state);

  if ( x >= scn.width || y >= scn.height ) return;

  float4 clrTotal = make_float4(0,0,0,0);
  int num_samples = 100;
  for (int i = 0; i < num_samples; ++i) {
    float3 hit = make_float3(NOHIT,NOHIT,NOHIT);
    float4 clr = make_float4(1,1,1,1);
    float3 norm;
    float randomx = (curand(&state) % 10)/10.0;
    float randomy = (curand(&state) % 10)/10.0;
    float3 rdir = normalize ( getViewRay ( (float(x)+randomx)/scn.width,
                                           (float(y)+randomy)/scn.height ) );

    // Ray march - trace a ray into GVDB and find the closest hit point
    rayCast ( SCN_SHADE, gvdb.top_lev, 0, scn.campos, rdir, hit, norm, clr, raySurfaceTrilinearBrick );

    if ( hit.z != NOHIT) {
      float3 lightdir = normalize ( scn.light_pos - hit );

      // Shading - custom look
      float3 eyedir = normalize ( scn.campos - hit );
      float3 R    = normalize ( reflect3 ( eyedir, norm ) );    // reflection vector
      float diffuse = max(0.0f, dot( norm, lightdir ));
      float refl    = min(1.0f, max(0.0f, R.y ));
      clr = diffuse*0.6 + refl * make_float4(1.0,0.3,0.7, 1.0);

    } else {
      clr = make_float4 ( 0.0, 0.0, 0.1, 1.0 );
    }
    clrTotal = clrTotal + clr / (float)num_samples;
  }
  outBuf [ y*scn.width + x ] = make_uchar4( clrTotal.x*255, clrTotal.y*255,
                                            clrTotal.z*255, 255 );
}

// nvcc evert-cuda.cu -o evert-cuda  -use_fast_math -O3  -Xcompiler "-Ofast -march=native"  -Xptxas "-O3 --verbose --warn-on-local-memory-usage --warn-on-spills"  &&  ./evert-cuda
#include <stdint.h>
#include <stdio.h>
#include <locale.h>
#define M_TAU  6.283185  // The (approximate) arclength of a circle of radius 1

// GPU general data!
#define EV_NGPUS     2
#define EV_GPU_MAIN  1
#define EV_IMG_DIR   "."  // "/mnt/tmp"  // "."
#define EV_EPSILON   0.001f
#define EV_RGB_BG    0x080808
#define EV_NLIGHTS   6  // The number of 2-faces of a cube!

// PIXEL SHADER data!
#define EV_NFRAMES     7  // If the number of frame is EVEN, then there's NO MIDDLE FRAME (and vv)! WARN! NFRAMES must be at least 2
#define EV_NSAMPLES    (1<<1)
#define EV_NBOUNCES    4
#define EV_IMG_W       (1920>>1)
#define EV_IMG_H       (1080>>1)
#define EV_CAM_FOV     (M_PI/3)   // 2: 90 fov, 3: 60 fov, 4: 45 fov, 6: 30 fov
#define EV_CAM_POS     { 0, 0,    5.0}
#define EV_CAM_DIR     {-0,-0.03,-1.0}
#define EV_CAM_ROT_YZ  0.0  // Camera rotation over the yz-plane
#define EV_CAM_ROT_ZX  0.0  // Camera rotation over the zx-plane
#define EV_CAM_ROT_XY  0.0  // Camera rotation over the xy-plane

// MESH shader data! @theta is the AZIMUTHAL parameter; @v is the POLAR parameter!
#define EV_NSTRIPS       8
#define EV_THETA_MIN     (0)
#define EV_PHI_MIN       (0 + EV_EPSILON)
#define EV_THETA_MAX     ((8./EV_NSTRIPS)*M_TAU)  // 8
#define EV_PHI_MAX       ((2./2)         *M_PI  - EV_EPSILON)   // 2
#define EV_THETA_NVERTS  (30*1*(EV_THETA_MAX-EV_THETA_MIN)/M_TAU*EV_NSTRIPS)
#define EV_PHI_NVERTS    (30*2*(EV_PHI_MAX  -EV_PHI_MIN)  /M_PI *2)
#define EV_RGB_FRONT     0xff9999  // 0xff6666
#define EV_RGB_BACK      0x5eaeff  // 0x1188ff

// STAGE times!
#define EV_CORRUGATE_TDEL    1.f
#define EV_PUSH_TDEL         2.f
#define EV_TWIST_TDEL        6.f
#define EV_UNPUSH_TDEL       2.f
#define EV_UNCORRUGATE_TDEL  1.f

#define EV_CORRUGATE_TINI    (0.f)
#define EV_PUSH_TINI         (EV_CORRUGATE_TINI+EV_CORRUGATE_TDEL)
#define EV_TWIST_TINI        (EV_PUSH_TINI     +EV_PUSH_TDEL)
#define EV_UNPUSH_TINI       (EV_TWIST_TINI    +EV_TWIST_TDEL)
#define EV_UNCORRUGATE_TINI  (EV_UNPUSH_TINI   +EV_UNPUSH_TDEL)

#define EV_TMIN  (EV_CORRUGATE_TINI)
#define EV_TMAX  (EV_CORRUGATE_TINI + EV_CORRUGATE_TDEL+EV_PUSH_TDEL+EV_TWIST_TDEL+EV_UNPUSH_TDEL+EV_UNCORRUGATE_TDEL)

// ----------------------------------------------------------------------------------------------------------------------------#
typedef uint8_t   u8;
typedef float     f32;
typedef int32_t   i32;
typedef uint32_t  u32;
typedef double    f64;
typedef uint64_t  u64;

// ----------------------------------------------------------------------------------------------------------------------------#
#include <time.h>
struct dt_t{
  f64 t0, t1;
};
f64  dt_abs(){  struct timespec tabs; clock_gettime(CLOCK_MONOTONIC,&tabs);  return tabs.tv_sec + 1e-9*tabs.tv_nsec;  }  // m_checksys(st,"clock_gettime");
f64  dt_del(dt_t* dt){  return dt->t1 - dt->t0;  }  // Get `relative time`, ie. a time delta between 2 absolute times! The time delta is returned in seconds, and its resolution is in nanoseconds!
void dt_ini(dt_t* dt){  dt->t0 = dt_abs();  }
void dt_end(dt_t* dt){  dt->t1 = dt_abs();  }

// ----------------------------------------------------------------------------------------------------------------------------#
#define cuda_check(){  cudaError_t err;  while((err=cudaGetLastError()) != cudaSuccess) printf("\x1b[91mFAIL\x1b[0m  \x1b[32mCUDA\x1b[0m  \x1b[32m%s\x1b[0m:\x1b[94mL%d\x1b[0m \x1b[35m%s\x1b[0m  \x1b[33m%s\x1b[0m \x1b[37m%s\x1b[0m\n", __FILE__,__LINE__,__func__, cudaGetErrorName(err),cudaGetErrorString(err));  }
#define m_divceilu(N, D)  (((N)%(D)) ? (N)/(D)+1 : (N)/(D))




// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @block */

// ----------------------------------------------------------------------------------------------------------------------------#
struct vec3{  // Just a simple 3D vector!
  union{  // Access the `vec3` using array notation of by specifying the name of a component!
    f32 data[3];
    struct{ f32 x0, x1, x2; };
  };
  __device__ __host__ vec3(){}
  __device__ __host__ vec3(f32 a0, f32 a1, f32 a2){  x0=a0; x1=a1; x2=a2;  }
  __device__ __host__ f32 operator[](int idx){       return data[idx];  }
};
__device__ __host__ vec3 operator*(f32   s, vec3  v){  return {s*v[0], s*v[1], s*v[2]};  }                 // Scalar multiplication!
__device__ __host__ vec3 operator+(vec3 v0, vec3 v1){  return {v0[0]+v1[0], v0[1]+v1[1], v0[2]+v1[2]};  }  // Vector addition!
__device__ __host__ vec3 operator-(vec3 v0, vec3 v1){  return {v0[0]-v1[0], v0[1]-v1[1], v0[2]-v1[2]};  }  // Vector subtraction!
__device__ __host__ vec3 operator*(vec3 v0, vec3 v1){  return {v0[0]*v1[0], v0[1]*v1[1], v0[2]*v1[2]};  }  // Vector multiplication!

__device__ __host__ f32  dot(vec3 v0, vec3 v1){  return v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2];  }  // Quite important for triangle intersection and a bit for the path tracer!
__device__ __host__ vec3 cross(vec3 v0, vec3 v1){  // Homology of R3: 0 --> 1 --> 2 --> 0 --> 1 --> 2 --> 0 --> ...
  return {v0[1]*v1[2] - v0[2]*v1[1],   // 0 --> 1 --> 2
          v0[2]*v1[0] - v0[0]*v1[2],   // 1 --> 2 --> 0
          v0[0]*v1[1] - v0[1]*v1[0]};  // 2 --> 0 --> 1
}
__device__ __host__ vec3 normalize(vec3 v){  return rsqrtf(dot(v,v)) * v;  }

// ----------------------------------------------------------------------------------------------------------------------------#
// The mighty quaternions! A real Clifford algebra (aka. a geometric algebra) related to:
// spinors, 3D rotations, the 3-sphere living in 4D, the gauge group SU(2) from quantum flavordynamics,
// the Whitehead tower of the orthogonal group O(3), the string group String(3), the fivebrane group Fivebrane(3), ...
struct quat{
  union{
    f32 data[4];
    struct{ f32 x0, x1, x2, x3; };
  };
  __device__ __host__ quat(){}
  __device__ __host__ quat(f32 a0, f32  a1, f32 a2, f32 a3){  x0=a0; x1=a1;   x2=a2;   x3=a3;    }
  __device__ __host__ quat(f32  s, vec3 v){                   x0=s;  x1=v[0]; x2=v[1]; x3=v[2];  }
  __device__ __host__ f32 operator[](int idx){                return data[idx];  }
};
__device__ __host__ quat operator*(quat q0, quat q1){  // The quaternion product is a sort of "twisted" product of 4D vectors
  return {q0[0]*q1[0] - q0[1]*q1[1] - q0[2]*q1[2] - q0[3]*q1[3],
          q0[0]*q1[1] + q0[1]*q1[0] + q0[2]*q1[3] - q0[3]*q1[2],
          q0[0]*q1[2] - q0[1]*q1[3] + q0[2]*q1[0] + q0[3]*q1[1],
          q0[0]*q1[3] + q0[1]*q1[2] - q0[2]*q1[1] + q0[3]*q1[0]};
}
__device__ __host__ quat conj(quat q){  return {q[0], -q[1], -q[2], -q[3]};  }  // The quaternion inverse of a quaternion `q` is just `conj(q) / quad(q)`, just like for complex numbers!

__device__ __host__ quat versor(f32 angle, vec3 dir){
  return {cosf(.5*angle), sinf(.5*angle)*normalize(dir)};
}
__device__ __host__ vec3 qrotl(vec3 v, quat versor){  // WARN! @versor must be a unit-quaternion!
  quat p_rot = versor * quat(0,v) * conj(versor);  // Left-conjugation by @versor! The quaternion-inverse of a unit-quaternion is its quaternion-conjugate!
  return {p_rot[1], p_rot[2], p_rot[3]};
}

// ----------------------------------------------------------------------------------------------------------------------------#
__forceinline__ __device__ f32  clamp01(f32  x){  return __saturatef(x);  }
__forceinline__ __device__ vec3 clamp01(vec3 v){  return {clamp01(v[0]), clamp01(v[1]), clamp01(v[2])};  }
__forceinline__ __device__ f32  rgb_gamma_decode(f32 channel){  return __powf(channel, 2.2/1);  }
__forceinline__ __device__ f32  rgb_gamma_encode(f32 channel){  return __powf(channel, 1/2.2);  }
__forceinline__ __device__ f32  rgb_u8_to_f32(   u8  channel){  return rgb_gamma_decode(channel/255.); }
__forceinline__ __device__ u8   rgb_f32_to_u8(   f32 channel){  return 255.*rgb_gamma_encode(channel) + .5; }
__forceinline__ __device__ vec3 bgr8u_to_rgb32f(u32 bgr8u){
  return {rgb_u8_to_f32((bgr8u>>0x10)&0xff),
          rgb_u8_to_f32((bgr8u>>0x08)&0xff),
          rgb_u8_to_f32((bgr8u>>0x00)&0xff)};
}
__forceinline__ __device__ u32 rgb32f_to_bgr8u(vec3 rgbf32){
  return (rgb_f32_to_u8(rgbf32[0])<<0x10) |
         (rgb_f32_to_u8(rgbf32[1])<<0x08) |
         (rgb_f32_to_u8(rgbf32[2])<<0x00);
}
__forceinline__ __device__ u32 rgb32f_to_rgb8u(vec3 rgbf32){
  return (rgb_f32_to_u8(rgbf32[0])<<0x00) |
         (rgb_f32_to_u8(rgbf32[1])<<0x08) |
         (rgb_f32_to_u8(rgbf32[2])<<0x10);
}

__forceinline__ __device__ f32 rand_f32(u32* seed0, u32* seed1){  // Random number generator from https://github.com/gz/rust-raytracer
  *seed0  = 36969*(*seed0&0xffff) + (*seed0>>0x10);
  *seed1  = 18489*(*seed1&0xffff) + (*seed1>>0x10);
  u32 val_u32 = 0x40000000 | (((*seed0<<0x10) + *seed1) & 0x007fffff);
  return .5f * (*(f32*)&val_u32) - 1.f;
}




// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// @block  Geometric data structures! Each geometric primitive needs its own intersection routine!

// ----------------------------------------------------------------------------------------------------------------------------#
struct light_t{
  vec3 vert0;  // Geometry/intersection data! vert0 IS the main vertex, edge01 IS vert1 - vert0, edge02 IS vert2 - vert0
  vec3 edge01;
  vec3 edge02;
  vec3 emission;  // Lighting/rendering data!
};
struct triangle_t{
  vec3 vert0;  // Geometry/intersection data! vert0 IS the main vertex, edge01 IS vert1 - vert0, edge02 IS vert2 - vert0
  vec3 edge01;
  vec3 edge02;
  u32  albedo_back;  // Lighting/rendering data! Albedo is the base color input, aka. a diffuse map... or something
  u32  albedo_front;
};
enum geom_type_t{ GEOM_UNKNOWN=0, GEOM_LIGHT, GEOM_TRIANGLE};

// ----------------------------------------------------------------------------------------------------------------------------#
// @section  Path tracing data structures!
struct ray_t{
  vec3 pos;  // Ray origin!
  vec3 dir;  // Ray direction!
};
struct intersect_t{  // We return this data structure upon hitting something when path tracing!
  f32 t;
  int front;  // Did we hit the front or the back?
};
struct hit_t{
  f32 t;      // The position of the hit in RAY COORDINATES. A ray is 1-dimensional, so its coordinates are 1-dimensional, too! Here we record *where* we hit the object!
  u32 idx;    // The object index, so that we know which object we hit!
  u32 type;   // What type of object did we hit, and in which mesh?
  int front;  // Did we hit the front or the back?
};




// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @block  EVERT code, originally by Nathaniel Thurston (Copyright 1995 Geometry Center, University of Minnesota)

What follows is a fork of Nathaniel Thurston's `evert` code, which implemented the Thurston sphere eversion for the movie Outside In, of meme-tastic fame.

The basic building blocks in `evert` are the so-called jets.
I have no idea what they're supposed to "be", but in `evert` they keep track of differential data associated to a scalar field.
As far as `evert` is concerned, they seem to have been used to implement automatic differentiation,
since partial derivatives are used to concoct the "figure eight" towards the end of the eversion code.

// ----------------------------------------------------------------------------------------------------------------------------#
                    Copyright (c) 1993

       The National Science and Technology Research Center for
        Computation and Visualization of Geometric Structures
                      (The Geometry Center)
                     University of Minnesota
                     1300 South Second Street
                  Minneapolis, MN  55454  USA

                  email: software@geom.umn.edu

The software distributed here is copyrighted as noted above.
It is free software and may be obtained via anonymous ftp from
ftp.geom.umn.edu.  It may be freely copied, modified, and
redistributed under the following conditions:

1. All copyright notices must remain intact in all files.

2. A copy of this file (COPYING) must be distributed along with any
   copies that you redistribute; this includes copies that you have
   modified, or copies of programs or other software products that
   include this software.

3. If you modify this software, you must include a notice giving the
   name of the person performing the modification, the date of
   modification, and the reason for such modification.

4. When distributing modified versions of this software, or other
   software products that include this software, you must provide
   notice that the original source code may be obtained as noted
   above.

5. There is no warranty or other guarantee of fitness for this
   software, it is provided solely "as is".  Bug reports or fixes may
   be sent to the email address above; the authors may or may not act
   on them as they desire.

If you use an image produced by this software in a publication or
presentation, we request that you credit the Geometry Center with a
notice such as the following:

  Figures 1, 2, and 5-300 were generated with software written at the
  Geometry Center, University of Minnesota.
*/

// ----------------------------------------------------------------------------------------------------------------------------#
/* @section  A 1-jet, aka. a first-order jet, aka. a scalar field (evaluated at some point) together with its 1st-order partial derivatives (evaluated at some point)!
We can think of a k-jet as an "augmented" floating-point number: the first entry in the struct is the value of the number,
and all the following entries are the partial derivatives up to order k.
Since we only need first-order partial derivatives, we'll only implement 1-jets.
*/
struct jet{
  f32 f;       // Scalar value of a 2D scalar field!
  f32 fu, fv;  // 1st-order partial derivatives of a 2D scalar field!
  __forceinline__ __device__ jet(){}
  __forceinline__ __device__ jet(f32 s){                  f=s; fu=0;  fv=0;   }
  __forceinline__ __device__ jet(f32 s, f32 su, f32 sv){  f=s; fu=su; fv=sv;  }
};
__forceinline__ __device__ jet operator+(jet x0, jet x1){  return {x0.f + x1.f, x0.fu + x1.fu, x0.fv + x1.fv};  }  // 1st-order partial derivatives of the addition    of two 2D scalar fields!
__forceinline__ __device__ jet operator-(jet x0, jet x1){  return {x0.f - x1.f, x0.fu - x1.fu, x0.fv - x1.fv};  }  // 1st-order partial derivatives of the subtraction of two 2D scalar fields!
__forceinline__ __device__ jet operator*(jet x0, jet x1){  // 1st-order partial derivatives of the product of two 2D scalar fields!
  return {x0.f *x1.f,
          x0.fu*x1.f + x0.f*x1.fu,
          x0.fv*x1.f + x0.f*x1.fv};
}
__forceinline__ __device__ jet operator%(jet x, f32 s){
  x.f = fmod(x.f, s);
  if(x.f<0) x.f += s;
  return x;
}
__forceinline__ __device__ jet operator^(jet x, f32 s){  // Derivatives of the n-th power?
  f32 f0 = powf(x.f, s);
  f32 f1 = x.f==0 ? 0 : s*f0/x.f;  // Avoid division by zero
  return {f0, f1*x.fu, f1*x.fv};
}
__forceinline__ __device__ jet operator/(jet x0, jet x1){  return x0 * (x1^-1);  }  // Derivatives of the quotient!

__forceinline__ __device__ jet ev_interpolate(jet x0, jet x1, jet t){  return (jet(1)-t)*x0 + t*x1;  }
__forceinline__ __device__ jet ev_partial_diff(jet x, int idx){        return {idx==0 ? x.fu : x.fv, 0,0};  }  // Keep the partial WRT u, or the partial WRT v?

__forceinline__ __device__ jet ev_cos(jet x){  f32 c=cosf(x.f);  f32 dc=-sinf(x.f);  return {c, dc*x.fu, dc*x.fv};  }  // Derivatives of the cosine of a scalar field!
__forceinline__ __device__ jet ev_sin(jet x){  f32 s=sinf(x.f);  f32 ds= cosf(x.f);  return {s, ds*x.fu, ds*x.fv};  }  // Derivatives of the sine of a scalar field!

// ----------------------------------------------------------------------------------------------------------------------------#
/* @section  A 3D vector of 1-jets!
If a k-jet can be interpreted as a "fat" floating-point number (a number augmented with differential data),
then a vector of k-jets is a vector of "fat" numbers: each entry in the vector is a "fat" number.

All the complicated bookkeeping needed to implement automatic differentiation happens at the level of k-jets,
so vectors of k-jets need only implement vector-specific stuff.

A 3D vector (of 1-jets) will represent a point in 3D space.
*/
struct vjet{
  jet x0, x1, x2;
};
__forceinline__ __device__ vjet operator*(jet   s, vjet v){   return {s*v.x0, s*v.x1, s*v.x2};  }                       // Scalar multiplication!
__forceinline__ __device__ vjet operator+(vjet v0, vjet v1){  return {v0.x0 + v1.x0, v0.x1 + v1.x1, v0.x2 + v1.x2};  }  // Vector addition!
__forceinline__ __device__ vjet operator-(vjet v0, vjet v1){  return {v0.x0 - v1.x0, v0.x1 - v1.x1, v0.x2 - v1.x2};  }  // Vector subtraction!

__forceinline__ __device__ jet  ev_dot(  vjet v0, vjet v1){  return v0.x0*v1.x0 + v0.x1*v1.x1 + v0.x2*v1.x2;  }
__forceinline__ __device__ vjet ev_cross(vjet v0, vjet v1){  // Homology of R3: 0 --> 1 --> 2 --> 0 --> 1 --> 2 --> 0 --> ...
  return {v0.x1*v1.x2 - v0.x2*v1.x1,   // 0 --> 1 --> 2
          v0.x2*v1.x0 - v0.x0*v1.x2,   // 1 --> 2 --> 0
          v0.x0*v1.x1 - v0.x1*v1.x0};  // 2 --> 0 --> 1
}
__forceinline__ __device__ vjet ev_normalize(vjet v){
  jet s = ev_dot(v,v);
  if(s.f>0) s = s^-.5;  // Avoid division by zero!
  else      s = jet(0);
  return s*v;
}

__forceinline__ __device__ vjet ev_interpolate( vjet v0, vjet v1, jet t){  return (jet(1)-t)*v0 + t*v1;  }
__forceinline__ __device__ vjet ev_partial_diff(vjet v,  int  idx){        return {ev_partial_diff(v.x0,idx), ev_partial_diff(v.x1,idx), ev_partial_diff(v.x2,idx)};  }

// ----------------------------------------------------------------------------------------------------------------------------#
/* @section  A quaternion made of 1-jets!
If a k-jet can be interpreted as a "fat" floating-point number (a number augmented with differential data),
and a vector of k-jets is a vector of "fat" numbers,
then a quaternion of 1-jets is a just quaternion... whose entries are not plain numbers, but "fat" numbers.

All the complicated bookkeeping needed to implement automatic differentiation happens at the level of k-jets,
so quaternions of k-jets need only implement quaternion-specific stuff.

We'll use quaternions to do rotations in 3D.
*/
struct qjet{
  jet x0, x1, x2, x3;
  __forceinline__ __device__ qjet(jet a0, jet  a1, jet a2, jet a3){  x0=a0; x1=a1;   x2=a2;   x3=a3;    }
  __forceinline__ __device__ qjet(jet  s, vjet v){                   x0=s;  x1=v.x0; x2=v.x1; x3=v.x2;  }
};
__forceinline__ __device__ qjet operator*(qjet q0, qjet q1){
  return {q0.x0*q1.x0 - q0.x1*q1.x1 - q0.x2*q1.x2 - q0.x3*q1.x3,
          q0.x0*q1.x1 + q0.x1*q1.x0 + q0.x2*q1.x3 - q0.x3*q1.x2,
          q0.x0*q1.x2 - q0.x1*q1.x3 + q0.x2*q1.x0 + q0.x3*q1.x1,
          q0.x0*q1.x3 + q0.x1*q1.x2 - q0.x2*q1.x1 + q0.x3*q1.x0};
}
__forceinline__ __device__ qjet ev_conj(qjet q){  return {q.x0, -1*q.x1, -1*q.x2, -1*q.x3};  }  // The quaternion-inverse of `q` is just `conj(q) / quad(q)`, just like for complex numbers!

__forceinline__ __device__ qjet ev_versor(jet angle, vjet dir){
  return {ev_cos(.5*angle), ev_sin(.5*angle)*ev_normalize(dir)};
}
__forceinline__ __device__ vjet ev_qrot3d(vjet v, qjet versor){
  qjet p_rot  = ev_conj(versor) * qjet(0,v) * versor;  // Right-conjugation by @versor! The quaternion-conjugate of a unit-quaternion is its quaternion-inverse!
  return {p_rot.x1, p_rot.x2, p_rot.x3};
}
__forceinline__ __device__ vjet ev_qrot_yz(vjet v, jet angle){  return ev_qrot3d(v, ev_versor(angle, {jet(1),jet(0),jet(0)})); }  // Rotation over the yz-plane
__forceinline__ __device__ vjet ev_qrot_zx(vjet v, jet angle){  return ev_qrot3d(v, ev_versor(angle, {jet(0),jet(1),jet(0)})); }  // Rotation over the zx-plane
__forceinline__ __device__ vjet ev_qrot_xy(vjet v, jet angle){  return ev_qrot3d(v, ev_versor(angle, {jet(0),jet(0),jet(1)})); }  // Rotation over the xy-plane

// ----------------------------------------------------------------------------------------------------------------------------#
// @section  Sphere parametrization and geometric deformations!
__forceinline__ __device__ vjet ev_sphere_arc(jet phi, f32 radius_x0, f32 radius_x1, f32 radius_x2){  // Trace out a meridian, since the horizontal angle is fixed!
  jet s0 = radius_x0 * ev_sin(jet(0,0,1)) * ev_sin(phi);  // Keep the horizontal angle constant, vary the vertical angle!
  jet s1 = radius_x1 * ev_cos(jet(0,0,1)) * ev_sin(phi);  // Keep the horizontal angle constant, vary the vertical angle!
  jet s2 = radius_x2 * ev_cos(phi);
  return {s0, s1, s2};
}

__forceinline__ __device__ jet ev_phi_deform0(jet phi){  // Map the (0..pi) interval to itself, but with some curvature!
  if(phi.f <= M_PI/2) return -2/M_PI*(phi^2) + 2*phi;
  else                return  2/M_PI*(phi^2) - 2*phi + jet(M_PI);
}
__forceinline__ __device__ jet ev_phi_deform1(jet phi){  // Map (0..xi) to (0..xi) with some curvature, and map (xi..pi) to (5xi..6xi) with some curvature!
  if(phi.f <= M_PI/2) return  2/M_PI*(phi^2);
  else                return -2/M_PI*(phi^2) + 4*phi + jet(M_PI);
}
__forceinline__ __device__ jet ev_phi_deform2(jet phi){
  if(phi.f > M_PI/2) phi = jet(M_PI) - phi;
  return -16/(M_PI*M_PI*M_PI)*(phi^3) + 12/(M_PI*M_PI)*(phi^2);
}


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @section  Thurston-eversion stages! A 3D vjet (of 1-jets, 2-jets, or 3-jets, or k-jets) always represents a point in R3!
So, if the output of a function is a vjet, then we can always render its output as a vertex in R3!
*/

// ----------------------------------------------------------------------------------------------------------------------------#
// Low-level stages!
__forceinline__ __device__ vjet ev_stage1(jet phi){  return ev_sphere_arc(phi,+1,+1,+1);  }
__forceinline__ __device__ vjet ev_stage2(jet phi){  return ev_interpolate(ev_sphere_arc(ev_phi_deform0(phi),+.9,+.9,-1), ev_sphere_arc(ev_phi_deform1(phi),+1,+1,+.5), ev_phi_deform2(phi));  }
__forceinline__ __device__ vjet ev_stage3(jet phi){  return ev_interpolate(ev_sphere_arc(ev_phi_deform0(phi),-.9,-.9,-1), ev_sphere_arc(ev_phi_deform1(phi),-1,+1,-.5), ev_phi_deform2(phi));  }
__forceinline__ __device__ vjet ev_stage4(jet phi){  return ev_sphere_arc(phi,-1,-1,-1);  }

// ----------------------------------------------------------------------------------------------------------------------------#
// Mid-level stages!
__forceinline__ __device__ vjet ev_scene12(jet phi, f32 t){  return ev_interpolate(ev_stage1(phi), ev_stage2(phi), jet(t));  }
__forceinline__ __device__ vjet ev_scene23(jet phi, f32 t){  // The heart of the TWIST stage! Notice the rotations here! =D
  t *= .5;
  f32  tt   = (phi.f<=M_PI/2) ? t : -t;
  vjet rot_xy = ev_qrot_xy(ev_sphere_arc(ev_phi_deform0(phi),+0.9,+0.9,-1.0), M_TAU*jet(tt));
  vjet rot_zx = ev_qrot_zx(ev_sphere_arc(ev_phi_deform1(phi),+1.0,+1.0,+0.5), M_TAU*jet(t));
  return ev_interpolate(rot_xy, rot_zx, ev_phi_deform2(phi));
}
__forceinline__ __device__ vjet ev_scene34(jet phi, f32 t){  return ev_interpolate(ev_stage3(phi), ev_stage4(phi), jet(t));  }

// ----------------------------------------------------------------------------------------------------------------------------#
// High-level stages!
__forceinline__ __device__ vjet ev_figure8(vjet w,vjet h, vjet bend, jet form, jet theta){  // At the end of the twisting phase, the corrugations have nearly become figure eights!
  theta = theta%1;
  jet height = 1 - ev_cos(2*M_TAU*theta);
  if(.25<theta.f && theta.f<.75) height = 4-height;
  height = .6*height;
  h      = h + (height*height)/(8*8) * bend;
  form   = 2*form - form*form;
  return ev_sin(2*M_TAU*theta)*w + ev_interpolate(2-2*ev_cos(M_TAU*theta), height, form)*h;
}

__forceinline__ __device__ vjet ev_add_figure8(vjet p, jet theta, jet phi, jet form, i32 nstrips){
  jet size = -0.2 * ev_phi_deform2(phi) * form;  // 0.2 is like a scale constant?

  vjet du = ev_normalize(ev_partial_diff(p,0));  // Is this the partial with respect to theta, or with respect to phi?
  vjet dv = ev_normalize(ev_partial_diff(p,1));  // Is this the partial with respect to theta, or with respect to phi?
  vjet h  = 1.0*size * ev_normalize(ev_cross(du,dv));
  vjet w  = 1.1*size * ev_normalize(ev_cross(h, du));  // The 1.1 factor gives more thickness/width to the corrugations?

  vjet bend = ev_partial_diff(size,0)/ev_partial_diff(phi,0) * du;
  vjet fig8 = ev_figure8(w,h, bend, form, (f32)nstrips/(f32)M_TAU*theta);
  return ev_qrot_xy(p+fig8, theta);
}


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// @section  Thurston-eversion phases!
// @theta goes from 0 to TAU. It describes a full circle on the xy-plane
// @phi   goes from 0 to PI.  It describes half-a-circle on the zx-plane
__device__ vjet ev_corrugate(  jet theta, jet phi, f32 t, i32 nstrips){  vjet p=ev_stage1( phi  );  return ev_add_figure8(p, theta,phi, jet(t)  *ev_phi_deform2(phi), nstrips);  }
__device__ vjet ev_push(       jet theta, jet phi, f32 t, i32 nstrips){  vjet p=ev_scene12(phi,t);  return ev_add_figure8(p, theta,phi, jet(1)  *ev_phi_deform2(phi), nstrips);  }
__device__ vjet ev_twist(      jet theta, jet phi, f32 t, i32 nstrips){  vjet p=ev_scene23(phi,t);  return ev_add_figure8(p, theta,phi, jet(1)  *ev_phi_deform2(phi), nstrips);  }
__device__ vjet ev_unpush(     jet theta, jet phi, f32 t, i32 nstrips){  vjet p=ev_scene34(phi,t);  return ev_add_figure8(p, theta,phi, jet(1)  *ev_phi_deform2(phi), nstrips);  }
__device__ vjet ev_uncorrugate(jet theta, jet phi, f32 t, i32 nstrips){  vjet p=ev_stage4( phi  );  return ev_add_figure8(p, theta,phi, jet(1-t)*ev_phi_deform2(phi), nstrips);  }


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// @section  Thurston-eversion geometry driver!
// @t     must be in the [0..1] interval!
// @theta can be in any interval, although its range should be at most TAU (unless you want to cover the sphere multiple times)! Eg. [0..TAU) is good. [-PI..PI) is good.
// @phi   must be in the (0..phi) interval, unless you want to hit those pesky singularities at the poles of the standard sphere parametrization!
__device__ void ev_quad(f32 t, f32 theta,f32 dtheta, f32 phi,f32 dphi, i32 nstrips, vjet* vert0_jet,vjet* vert1_jet,vjet* vert2_jet,vjet* vert3_jet){
  f32 t_ev = t;
  if(t_ev-EV_EPSILON < EV_CORRUGATE_TINI+EV_CORRUGATE_TDEL){  // WARN! For some reason we need to subtract the time by EV_EPSILON?
    *vert0_jet = ev_corrugate(jet(theta + 0*dtheta), jet(phi + 0*dphi, 1,0), (t_ev-EV_CORRUGATE_TINI)/EV_CORRUGATE_TDEL, nstrips);
    *vert1_jet = ev_corrugate(jet(theta + 1*dtheta), jet(phi + 0*dphi, 1,0), (t_ev-EV_CORRUGATE_TINI)/EV_CORRUGATE_TDEL, nstrips);
    *vert2_jet = ev_corrugate(jet(theta + 1*dtheta), jet(phi + 1*dphi, 1,0), (t_ev-EV_CORRUGATE_TINI)/EV_CORRUGATE_TDEL, nstrips);
    *vert3_jet = ev_corrugate(jet(theta + 0*dtheta), jet(phi + 1*dphi, 1,0), (t_ev-EV_CORRUGATE_TINI)/EV_CORRUGATE_TDEL, nstrips);
  }else if(t_ev-EV_EPSILON < EV_PUSH_TINI+EV_PUSH_TDEL){
    *vert0_jet = ev_push(jet(theta + 0*dtheta), jet(phi + 0*dphi, 1,0), (t_ev-EV_PUSH_TINI)/EV_PUSH_TDEL, nstrips);
    *vert1_jet = ev_push(jet(theta + 1*dtheta), jet(phi + 0*dphi, 1,0), (t_ev-EV_PUSH_TINI)/EV_PUSH_TDEL, nstrips);
    *vert2_jet = ev_push(jet(theta + 1*dtheta), jet(phi + 1*dphi, 1,0), (t_ev-EV_PUSH_TINI)/EV_PUSH_TDEL, nstrips);
    *vert3_jet = ev_push(jet(theta + 0*dtheta), jet(phi + 1*dphi, 1,0), (t_ev-EV_PUSH_TINI)/EV_PUSH_TDEL, nstrips);
  }else if(t_ev-EV_EPSILON < EV_TWIST_TINI+EV_TWIST_TDEL){
    *vert0_jet = ev_twist(jet(theta + 0*dtheta), jet(phi + 0*dphi, 1,0), (t_ev-EV_TWIST_TINI)/EV_TWIST_TDEL, nstrips);
    *vert1_jet = ev_twist(jet(theta + 1*dtheta), jet(phi + 0*dphi, 1,0), (t_ev-EV_TWIST_TINI)/EV_TWIST_TDEL, nstrips);
    *vert2_jet = ev_twist(jet(theta + 1*dtheta), jet(phi + 1*dphi, 1,0), (t_ev-EV_TWIST_TINI)/EV_TWIST_TDEL, nstrips);
    *vert3_jet = ev_twist(jet(theta + 0*dtheta), jet(phi + 1*dphi, 1,0), (t_ev-EV_TWIST_TINI)/EV_TWIST_TDEL, nstrips);
  }else if(t_ev-EV_EPSILON < EV_UNPUSH_TINI+EV_UNPUSH_TDEL){
    *vert0_jet = ev_unpush(jet(theta + 0*dtheta), jet(phi + 0*dphi, 1,0), (t_ev-EV_UNPUSH_TINI)/EV_UNPUSH_TDEL, nstrips);
    *vert1_jet = ev_unpush(jet(theta + 1*dtheta), jet(phi + 0*dphi, 1,0), (t_ev-EV_UNPUSH_TINI)/EV_UNPUSH_TDEL, nstrips);
    *vert2_jet = ev_unpush(jet(theta + 1*dtheta), jet(phi + 1*dphi, 1,0), (t_ev-EV_UNPUSH_TINI)/EV_UNPUSH_TDEL, nstrips);
    *vert3_jet = ev_unpush(jet(theta + 0*dtheta), jet(phi + 1*dphi, 1,0), (t_ev-EV_UNPUSH_TINI)/EV_UNPUSH_TDEL, nstrips);
  }else if(t_ev-EV_EPSILON < EV_UNCORRUGATE_TINI+EV_UNCORRUGATE_TDEL){
    *vert0_jet = ev_uncorrugate(jet(theta + 0*dtheta), jet(phi + 0*dphi, 1,0), (t_ev-EV_UNCORRUGATE_TINI)/EV_UNCORRUGATE_TDEL, nstrips);
    *vert1_jet = ev_uncorrugate(jet(theta + 1*dtheta), jet(phi + 0*dphi, 1,0), (t_ev-EV_UNCORRUGATE_TINI)/EV_UNCORRUGATE_TDEL, nstrips);
    *vert2_jet = ev_uncorrugate(jet(theta + 1*dtheta), jet(phi + 1*dphi, 1,0), (t_ev-EV_UNCORRUGATE_TINI)/EV_UNCORRUGATE_TDEL, nstrips);
    *vert3_jet = ev_uncorrugate(jet(theta + 0*dtheta), jet(phi + 1*dphi, 1,0), (t_ev-EV_UNCORRUGATE_TINI)/EV_UNCORRUGATE_TDEL, nstrips);
  }
}




// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// @block  CUDA kernels for doing sexy stuff to the mesh

// CUDA kernel for the lights!
__global__ void ker_lights_init(light_t* lights){
  f32 z = vec3(EV_CAM_POS).x2 + EV_EPSILON;
  f32 x = 1024;
  lights[0] = {{-x/4, -2.2, -x/4}, {+x, 0, 0}, { 0, 0,+x}, {1.4,1.4,1.8}};  // Bottom face?
  lights[1] = {{-x/4, +1.8, -x/4}, {+x, 0, 0}, { 0, 0,+x}, {1.4,1.4,1.8}};  // Top face?
  lights[2] = {{-3.7, -x/4, +x/4}, { 0,+x, 0}, { 0, 0,-x}, {1.4,1.4,1.8}};  // Left face?
  lights[3] = {{+3.7, +x/4, -x/4}, { 0, 0,+x}, { 0,-x, 0}, {1.4,1.4,1.8}};  // Right face?
  lights[4] = {{+x/4, +x/4,   +z}, { 0,-x, 0}, {-x, 0, 0}, {1.4,1.4,1.8}};  // Front face?
  lights[5] = {{-x/4, -x/4,   -2}, {+x, 0, 0}, { 0,+x, 0}, bgr8u_to_rgb32f(EV_RGB_BG)};  // Back face!
}


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @section  This is the "mesh driver", ie. it takes Nathaniel Thurton's jet/evert stuff and actually creates the sphere eversion animation!
It creates vertex coordinates OUT OF THIN AIR (ie. out of kernel coordinates), per FRAME! How sexy is that? 0.5M triangles in 0.5ms!
*/
__global__ void ker_mesh_shader(f32 t, quat rot, u32 theta_nverts, u32 phi_nverts, triangle_t* triangles){
  u32 x       = blockIdx.x*blockDim.x + threadIdx.x;
  u32 y       = blockIdx.y*blockDim.y + threadIdx.y;
  u32 thr_idx = (blockIdx.y*gridDim.x + blockIdx.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x;  // Global thread index, see richiesams blogspot!

  // ----------------------------------------------------------------------------------------------------------------------------#
  f32 theta_gap = +.06f;  // +.06f  -.01f
  f32 phi_gap   = +.06f;  // +.06f  -.01f
  i32 theta_idx = x;
  i32 phi_idx   = y;
  f32 dtheta    = (EV_THETA_MAX-EV_THETA_MIN) / theta_nverts;  // "Vanilla" delta-theta!
  f32 dphi      = (EV_PHI_MAX  -EV_PHI_MIN)   / phi_nverts;    // "Vanilla" delta-phi!

  f32 theta      = dtheta*(f32)theta_idx + EV_THETA_MIN;  // Now theta is in [0 .. theta_max)
  f32 phi        = dphi  *(f32)phi_idx   + EV_PHI_MIN;    // Now phi   is in (0 .. phi_max)
  f32 dtheta_gap = (EV_THETA_MAX-EV_THETA_MIN) / (theta_nverts + theta_gap*theta_nverts);  // Delta-theta w/ a gap!
  f32 dphi_gap   = (EV_PHI_MAX  -EV_PHI_MIN)   / (phi_nverts   + phi_gap  *phi_nverts);    // Delta-phi   w/ a gap!

  vjet vert0_jet, vert1_jet, vert2_jet, vert3_jet;  ev_quad(t, theta,dtheta_gap, phi,dphi_gap, EV_NSTRIPS, &vert0_jet,&vert1_jet,&vert2_jet,&vert3_jet);

  // ----------------------------------------------------------------------------------------------------------------------------#
  vec3 vert0 = qrotl(vec3(vert0_jet.x0.f, vert0_jet.x1.f, vert0_jet.x2.f), rot);
  vec3 vert1 = qrotl(vec3(vert1_jet.x0.f, vert1_jet.x1.f, vert1_jet.x2.f), rot);
  vec3 vert2 = qrotl(vec3(vert2_jet.x0.f, vert2_jet.x1.f, vert2_jet.x2.f), rot);
  vec3 vert3 = qrotl(vec3(vert3_jet.x0.f, vert3_jet.x1.f, vert3_jet.x2.f), rot);

  vec3 color0  = bgr8u_to_rgb32f(EV_RGB_FRONT);  // sin(theta): as `theta` goes from 0 to TAU, `sin(theta)` goes from 0 to 0
  vec3 color1  = bgr8u_to_rgb32f(EV_RGB_BACK);   // sin(2*phi): as `phi`   goes from 0 to PI,  `sin(2*phi)` goes from 0 to 0
  vec3 dcolor0 = .2f * vec3(0,0,(sinf(theta)+1)/2);
  vec3 dcolor1 = .3f * vec3((sinf(theta)+1)/2,0,0);

  triangle_t triangle;
  triangle.albedo_back  = rgb32f_to_bgr8u(clamp01(color1 + dcolor1));
  triangle.albedo_front = rgb32f_to_bgr8u(clamp01(color0 + dcolor0));
  triangle.vert0=vert0;  triangle.edge01=vert1-vert0;  triangle.edge02=vert3-vert0;  triangles[2*thr_idx+0]=triangle;
  triangle.vert0=vert2;  triangle.edge01=vert3-vert2;  triangle.edge02=vert1-vert2;  triangles[2*thr_idx+1]=triangle;
}




// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @block  CUDA renderer: a path tracer!

This is a CUDA path tracer, originally by Sam Lapere.
I just stole it and repurposed it.

https://github.com/straaljager/GPU-path-tracing-tutorial-2
*/

// ----------------------------------------------------------------------------------------------------------------------------#
// CUDA device code for the geometry intersection, used when path tracing!
__forceinline__ __device__ intersect_t pt_triangle_intersect(triangle_t self, ray_t ray){  // Intersect this geometric primitive with a ray! Return the distance, or 0 if there's no hit!
  if(self.edge01[0]==0.f && self.edge01[1]==0.f && self.edge01[2]==0.f) return {0.f,0};  // This allows us to have "trivial" primitives in the mesh and not break the path tracer!
  vec3 ray_dir = ray.dir;
  vec3 edge01  = self.edge01;
  vec3 edge02  = self.edge02;
  vec3 op      = ray.pos - self.vert0;
  vec3 pvec    = cross(ray_dir,edge02);
  f32  det     = __fdividef(1.f, dot(edge01,pvec));  // CUDA intrinsic!  // I think __fdividef(1/x) is FASTER than __frcp_r*() for computing multiplicative inverses / reciprocals!
  f32  u       = det * dot(op,pvec);       if(u<0.f || u  >1.f) return {0.f,0};  // No intersection! Early exit DOES help!
  vec3 qvec    = cross(op,edge01);
  f32  v       = det * dot(ray_dir,qvec);  if(v<0.f || u+v>1.f) return {0.f,0};  // No intersection!
  f32  t       = det * dot(edge02,qvec);
  return {t, det>0.f};
}
__forceinline__ __device__ vec3 pt_triangle_normal(triangle_t self, vec3 x){  // A triangle has to curvature, so it's normal vector field is a CONSTANT vector field: its value is constant!
  return normalize(cross(self.edge01,self.edge02));  // Cross product of two triangle edges yields a vector orthogonal to the triangle plane! Weee! A normal MUST be a unit vector!
}

__forceinline__ __device__ intersect_t pt_light_intersect(light_t self, ray_t ray){  // Intersect this geometric primitive with a ray! Return the distance, or 0 if there's no hit!
  return pt_triangle_intersect({self.vert0, self.edge01, self.edge02}, ray);
}
__forceinline__ __device__ vec3 pt_light_normal(light_t self, vec3 x){  // A triangle has to curvature, so it's normal vector field is a CONSTANT vector field: it's value is constant!
  return normalize(cross(self.edge01,self.edge02));  // Cross product of two triangle edges yields a vector orthogonal to the triangle plane! Weee! A normal MUST be a unit vector!
}

// ----------------------------------------------------------------------------------------------------------------------------#
__forceinline__ __device__ hit_t pt_scene_intersect(ray_t ray, u32 nlights,light_t* lights, u32 ntriangles,triangle_t* triangles){
  hit_t hit = {t:1e38f, idx:0, type:GEOM_UNKNOWN, front:-1};  // @pos is ray coordinate of the closest intersection

  for(int i=0; i<nlights; ++i){
    intersect_t intersect = pt_light_intersect(lights[i], ray);
    f32         t         = intersect.t;  if(t<EV_EPSILON || t>hit.t) continue;
    hit.t                 = t;
    hit.idx               = i;
    hit.type              = GEOM_LIGHT;
  }  // Record the position of the closest intersection point in RAY COORDINATES (which are 1-dimensional, so you need a single number), and also the ID of the object in question

  for(int i=0; i<ntriangles; ++i){
    intersect_t intersect = pt_triangle_intersect(triangles[i], ray);
    f32         t         = intersect.t;  if(t<EV_EPSILON || t>hit.t) continue;
    hit.t                 = t;
    hit.idx               = i;
    hit.type              = GEOM_TRIANGLE;
    hit.front             = intersect.front;
  }  // Record the position of the closest intersection point in RAY COORDINATES (which are 1-dimensional, so you need a single number), and also the ID of the object in question

  return hit;
}

// ----------------------------------------------------------------------------------------------------------------------------#
__forceinline__ __device__ vec3 pt_normal_out(vec3 normal, vec3 ray_dir){
  return dot(normal,ray_dir)<0 ? normal : -1*normal;  // "Outwards" normal, to create a "bounce"!
}

// Sample a random direction on the dome/hemisphere around the hitpoint base on the normal at that point!
__forceinline__ __device__ vec3 pt_dome_randdir(vec3 normal_out, uint* seed_x, uint* seed_y){
  // Compute local orthonormal basis/basis uvw at hitpoint, to compute the (random) ray direction.
  // 1st vector is normal at hitpoint, 2nd vector is orthogonal to 1st, 3rd vector is orthogonal to first others
  vec3 basis_w = normal_out;
  vec3 axis    = fabs(basis_w[0])<.1f ? vec3(1,0,0) : vec3(0,1,0);
  vec3 basis_u = normalize(cross(axis, basis_w));  // We shouldn't need to normalize this, but, if we don't, then we introduce artifacts!
  vec3 basis_v = cross(basis_w, basis_u);          // Right-handed uvw-basis! The homology is: u -> v -> w -> u -> ...

  // All our geometric primitives (just triangles) are diffuse, which reflect light uniformly in all directions!
  // Generate random direction in hemisphere above hitpoint (see "Realistic Ray Tracing", P. Shirley)
  f32 rand_tau  = rand_f32(seed_x,seed_y) * M_TAU;  // Get random number on unit circle for azimuth
  f32 rand_one  = rand_f32(seed_x,seed_y);          // Get random number for elevation
  f32 rand_sqrt = sqrtf(rand_one);  // No FAST intrinsic for sqrt?

  f32 cos_tau, sin_tau; __sincosf(rand_tau, &sin_tau,&cos_tau);
  return cos_tau*rand_sqrt*basis_u + sin_tau*rand_sqrt*basis_v + sqrtf(1.f-rand_one)*basis_w;  // Random ray direction on the hemisphere/dome around a point! Cosine-weighted importance sampling, favors ray directions closer to normal direction!
}

// ----------------------------------------------------------------------------------------------------------------------------#
// Here we solve the rendering equation: outgoing_radiance (at x)  ==  emitted_radiance (at x) + reflected_radiance (at x).
// Reflected radiance is sum/integral of incoming radiance from all directions in hemisphere above point, multiplied by reflectance function of material (BRDF) and cosine incident angle
__device__ vec3 pt_radiance_integral(ray_t ray, uint* seed_x,uint* seed_y, u32 nlights,light_t* lights, u32 ntriangles,triangle_t* triangles){
  vec3 rgb    = vec3(0,0,0);  // This will integrate/sum/accumulate the color over all bounces!
  vec3 fade   = vec3(1,1,1);
  vec3 rgb_bg = bgr8u_to_rgb32f(EV_RGB_BG);

  for(int bounce=0; bounce<EV_NBOUNCES; ++bounce){  // Iteration up to N bounces: replaces recursion in CPU code!
    hit_t hit     = pt_scene_intersect(ray, nlights,lights, ntriangles,triangles);  if(hit.t==1e38f) return vec3(0,0,0);  // No intersection! Return black!
    vec3  hit_pos = ray.pos + hit.t*ray.dir;  // @hit_pos is the hit position in WORLD COORDINATES! @hit.t is the hit position in RAY COORDINATES!

    // ----------------------------------------------------------------
    vec3 obj_normal, obj_rgb, obj_emi;
    switch(hit.type){  // Retrieve the geometric data of the object we hit!
      case GEOM_LIGHT:{
        light_t obj = lights[hit.idx];
        obj_normal  = pt_light_normal(obj, hit_pos);
        obj_rgb     = vec3(0,0,0);
        obj_emi     = obj.emission;
        }break;
      case GEOM_TRIANGLE:{
        triangle_t obj = triangles[hit.idx];
        obj_normal     = pt_triangle_normal(obj, hit_pos);
        obj_rgb        = hit.front ? bgr8u_to_rgb32f(obj.albedo_front) : bgr8u_to_rgb32f(obj.albedo_back);
        obj_emi        = vec3(0,0,0);
        }break;
    }
    rgb = rgb + fade*obj_emi;  // Add emission of current object to accumulated color (first term in rendering equation sum)

    // ----------------------------------------------------------------
    vec3 obj_normal_out = pt_normal_out(obj_normal, ray.dir);  // "Outwards" normal, to create a "bounce"!
    vec3 dome_dir       = pt_dome_randdir(obj_normal_out, seed_x,seed_y);

    fade    = dot(obj_normal_out, dome_dir) * obj_rgb * fade;  // 0) Integrate/sum/accumulate the fade! Weigh light/color energy using cosine of angle between normal and incident light!
    ray.pos = hit_pos + EV_EPSILON*obj_normal_out;  // 1) Launch a new raw starting by "bouncing" it from the object! Offset ray position slightly to prevent self intersection
    ray.dir = dome_dir;  // "Bounce" the ray from the surface at the hit position, oriented by the surface normal!
  }

  return rgb;
}

// ----------------------------------------------------------------------------------------------------------------------------# Map a CUDA thread to each pixel!
__global__ void ker_pixel_shader(u32 img_w,u32 img_h, u32 img_w_min,u32 img_w_max,u32 img_h_min,u32 img_h_max, u32* img_data, u32 nlights,light_t* lights, u32 ntriangles,triangle_t* triangles, f32 cam_fov,vec3 cam_pos,vec3 cam_dir,quat cam_rot, u32 seed){
  u32  px_x      = blockIdx.x*blockDim.x + threadIdx.x;
  u32  px_y      = blockIdx.y*blockDim.y + threadIdx.y;  if(px_x>=(img_w_max-img_w_min) || px_y>=(img_h_max-img_h_min)) return;
  u32  seed_x    = px_x + seed;
  u32  seed_y    = px_y + seed;

  // ----------------------------------------------------------------
  cam_dir        = qrotl(cam_dir, cam_rot);
  vec3 cam_dir_x = qrotl((.5*cam_fov) * vec3((f32)img_w/img_h, 0, 0), cam_rot);  // Cam ray is directed at the lower-left corner of the screen!
  vec3 cam_dir_y =       (.5*cam_fov) * normalize(cross(cam_dir,cam_dir_x));           // Cam ray is directed at the lower-left corner of the screen!

  // ----------------------------------------------------------------
  vec3 px_rgb = vec3(0,0,0);  // Final pixel color! Init to zero for each pixel!
  for(int sample=0; sample<EV_NSAMPLES; ++sample){  // Samples per pixel! Camera rays are pushed forward to start in interior
    f32   cam_dx  = (px_x + rand_f32(&seed_x,&seed_y)) / img_w - .5;
    f32   cam_dy  = (px_y + rand_f32(&seed_x,&seed_y)) / img_h - .5 + (f32)img_h_min/img_h;
    vec3  px_pos = cam_pos;
    vec3  px_dir = cam_dir + cam_dx*cam_dir_x + cam_dy*cam_dir_y;
    ray_t px_ray = {px_pos, normalize(px_dir)};
    px_rgb       = px_rgb + 1.f/EV_NSAMPLES * pt_radiance_integral(px_ray, &seed_x,&seed_y, nlights,lights, ntriangles,triangles);
  }

  img_data[px_y*img_w + px_x] = rgb32f_to_rgb8u(clamp01(px_rgb));
}




// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// @block  Multi-GPU computing data structure!
struct gpu_work_t{
  // General parameters!
  u32  id;   // Device ID!
  f32  t;    // Time!
  quat rot;  // Rotation quaternion!

  // Mesh shader stuff!
  u32         theta_nverts, phi_nverts;
  u32         nlights;
  u32         ntriangles;
  light_t*    lights_gpu;
  triangle_t* triangles_gpu;

  // Pixel shader stuff!
  u32  img_w, img_h;
  u32  img_w_min, img_w_max;
  u32  img_h_min, img_h_max;
  u32  img_tile_nelems;
  u32  img_tile_stride;
  u32* img_tile_gpu;
  u32* img_full_cpu;
};

gpu_work_t* gpu_work_init(u32 gpu_id, u32 img_w,u32 img_h, u32 img_w_min,u32 img_w_max,u32 img_h_min,u32 img_h_max, u32 img_tile_stride){
  gpu_work_t* gpu  = (gpu_work_t*)malloc(sizeof(gpu_work_t));
  gpu->id          = gpu_id;
  cudaSetDevice(gpu->id);  cuda_check();

  // ---------------------------------------------------------------- Mesh shader parameterss
  gpu->theta_nverts = ceilf(EV_THETA_NVERTS);
  gpu->phi_nverts   = ceilf(EV_PHI_NVERTS);
  gpu->nlights      = EV_NLIGHTS;
  gpu->ntriangles   = gpu->theta_nverts * gpu->phi_nverts * 2;

  // ---------------------------------------------------------------- Pixel shader parameters!
  gpu->img_w           = img_w;
  gpu->img_h           = img_h;
  gpu->img_w_min       = img_w_min;
  gpu->img_w_max       = img_w_max;
  gpu->img_h_min       = img_h_min;
  gpu->img_h_max       = img_h_max;
  gpu->img_tile_nelems = (img_w_max-img_w_min) * (img_h_max-img_h_min);
  gpu->img_tile_stride = img_tile_stride;

  // ---------------------------------------------------------------- Mesh shader buffers!
  cudaMalloc(&gpu->lights_gpu,    sizeof(light_t)   *gpu->nlights);
  cudaMalloc(&gpu->triangles_gpu, sizeof(triangle_t)*gpu->ntriangles);

  // ---------------------------------------------------------------- Pixel shader buffers!
  cudaMalloc(&gpu->img_tile_gpu, sizeof(u32)*gpu->img_tile_nelems);
  if(gpu->id==EV_GPU_MAIN)  cudaMallocHost(&gpu->img_full_cpu, sizeof(u32)*gpu->img_w*gpu->img_h);
  cuda_check();
  return gpu;
}

void gpu_work_free(gpu_work_t* gpu){
  cudaSetDevice(gpu->id);

  cudaFree(gpu->triangles_gpu);
  cudaFree(gpu->lights_gpu);

  cudaFree(gpu->img_tile_gpu);
  if(gpu->id==EV_GPU_MAIN)  cudaFreeHost(gpu->img_full_cpu);
  cudaDeviceReset();  cuda_check();
  free(gpu);
}

void gpu_sync(gpu_work_t* gpu){  // Always sync (only) stream zero
  cudaSetDevice(gpu->id);
  cudaStreamSynchronize(0);  cuda_check();
}

void gpu_mesh_shader(gpu_work_t* gpu){
  cudaSetDevice(gpu->id);
  dim3 block_dim = {1,1,1};  // Launch `block_dim.x * block_dim.y * block_dim.z` nthreads per block! So, `32 * 32 * 1` nthreads per block! Max nthreads per block on Titan V is 1024!
  dim3 grid_dim  = {m_divceilu(gpu->theta_nverts,block_dim.x), m_divceilu(gpu->phi_nverts,block_dim.y), 1};  // Launch ` grid_dim.x *  grid_dim.y *  grid_dim.z` nblocks  per grid!
  ker_lights_init<<<1,1>>>(gpu->lights_gpu);  cuda_check();
  ker_mesh_shader<<<grid_dim,block_dim>>>(gpu->t,gpu->rot, gpu->theta_nverts,gpu->phi_nverts, gpu->triangles_gpu);  cuda_check();
}

void gpu_pixel_shader(gpu_work_t* gpu, u32* img_cpu, u32 seed){
  quat cam_rot_yz = versor(EV_CAM_ROT_YZ, vec3(1,0,0));
  quat cam_rot_zx = versor(EV_CAM_ROT_ZX, vec3(0,1,0));
  quat cam_rot_xy = versor(EV_CAM_ROT_XY, vec3(0,0,1));
  quat cam_rot    = cam_rot_xy * cam_rot_zx * cam_rot_yz;

  cudaSetDevice(gpu->id);
  dim3 block_dim = {8,8,1};
  dim3 grid_dim  = {m_divceilu((gpu->img_w_max-gpu->img_w_min), block_dim.x), m_divceilu((gpu->img_h_max-gpu->img_h_min), block_dim.y), 1};
  ker_pixel_shader<<<grid_dim,block_dim>>>(gpu->img_w,gpu->img_h, gpu->img_w_min,gpu->img_w_max, gpu->img_h_min,gpu->img_h_max, gpu->img_tile_gpu, gpu->nlights,gpu->lights_gpu, gpu->ntriangles,gpu->triangles_gpu, EV_CAM_FOV,vec3(EV_CAM_POS),normalize(EV_CAM_DIR),cam_rot, seed);  cuda_check();
  cudaMemcpyAsync(img_cpu + gpu->img_tile_stride, gpu->img_tile_gpu, sizeof(u32)*gpu->img_tile_nelems, cudaMemcpyDeviceToHost, 0);  cuda_check();  // Default stream!
}

// ----------------------------------------------------------------------------------------------------------------------------#
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

void gpu_img_save(gpu_work_t* gpu, const char* img_dir, i32 frame){  // PPM+fwrite u32 saves 1920x1024 @ 60ms! PPM+fwrite float3 saves 1920x1024 @ 150ms! PNG+fwrite saves 1920x1024 @ 700ms!
  char img_path[PATH_MAX]={}; snprintf(img_path,PATH_MAX-1, "%s/%04d.ppm",img_dir,frame);
  char header_txt[256]={};
  int  header_nbytes = snprintf(header_txt,sizeof(header_txt)-1,"P6\n%d %d\n%d\n", gpu->img_w, gpu->img_h, 255);
  u32  pixel_nbytes  = 3*gpu->img_w*gpu->img_h;

  // open/ftruncate/mmap/close/memcpy/munmap 1920x1080: 7ms!  fwrite: 80ms!  write: 1380ms!
  int fd        = open(img_path, O_RDWR|O_CREAT|O_TRUNC, 0b110100100);  if(fd==-1){ printf("\x1b[31mWARN\x1b[0m  Can't \x1b[35mopen\x1b[0m \x1b[33m%s\x1b[0m! Discarding rendered image \x1b[31m=(\x1b[0m  ",img_path); return; }  // O_RDONLY O_WRONLY O_RDWR
  int st        = ftruncate(fd, header_nbytes+pixel_nbytes);            if(st==-1)  printf("\x1b[31mWARN\x1b[0m  Can't \x1b[35mftruncate\x1b[0m \x1b[33m%s\x1b[0m!  ",img_path);
  u8* data8_dst = (u8*)mmap(NULL, header_nbytes+pixel_nbytes, PROT_READ|PROT_WRITE,MAP_SHARED, fd,0);  if(data8_dst==MAP_FAILED) printf("\x1b[31mWARN\x1b[0m  Can't \x1b[35mmmap\x1b[0m \x1b[33m%s\x1b[0m!  ",img_path);
  st=close(fd);  if(st==-1) printf("\x1b[31mWARN\x1b[0m  Can't close file descriptor!");  // `mmap` adds an extra reference to the file associated with the file descriptor which is not removed by a subsequent `close` on that file descriptor!

  memcpy(data8_dst, header_txt, header_nbytes);
  data8_dst += header_nbytes;

  i32  height     = gpu->img_h;  // Caching these guys takes us from 7ms to 5ms!
  i32  width      = gpu->img_w;
  u32* data32_src = gpu->img_full_cpu;
  for(i32 i=height; 0<i; --i){  // The vertical index iterates backwards! =D
    for(i32 j=0; j<width; ++j){
      i32 lidx = (i-1)*width + j;
      memcpy(data8_dst + 3*lidx, data32_src + lidx, 3);  // memcpy 1920x1080: 6-7ms, single-byte addressing 1920x1080: 7-8ms
    }
  }

  munmap(data8_dst, header_nbytes+pixel_nbytes);
}

// ----------------------------------------------------------------------------------------------------------------------------#
void gpus_render_to_disk(gpu_work_t** gpus){  // Driver of GPU work!
  u64 nintersections = (u64)EV_IMG_W*EV_IMG_H * EV_NSAMPLES*EV_NBOUNCES * gpus[EV_GPU_MAIN]->ntriangles;
  u32 seed = time(NULL);
  f32 t    = EV_TMIN;
  f32 dt   = (EV_TMAX-EV_TMIN) / max(1,EV_NFRAMES-1);
  f64 spf;  dt_t tdel;
  putchar(0x0a);
  printf("\x1b[94mimg_w\x1b[0m     \x1b[0m%d\x1b[0m\n", gpus[EV_GPU_MAIN]->img_w);
  printf("\x1b[35mimg_h\x1b[0m     \x1b[0m%d\x1b[0m\n", gpus[EV_GPU_MAIN]->img_h);
  printf("\x1b[31mnframes\x1b[0m   \x1b[0m%d\x1b[0m\n", EV_NFRAMES);
  printf("\x1b[32mnsamples\x1b[0m  \x1b[0m%d\x1b[0m\n", EV_NSAMPLES);
  printf("\x1b[94mnbounces\x1b[0m  \x1b[0m%d\x1b[0m\n", EV_NBOUNCES);

  putchar(0x0a);
  printf("\x1b[32mimg\x1b[0m dir                   \x1b[33m%s\x1b[0m\n",   EV_IMG_DIR);
  printf("\x1b[32mtriangles\x1b[0m nelems          \x1b[94m%'u\x1b[0m\n",  gpus[EV_GPU_MAIN]->ntriangles);
  printf("\x1b[32mtriangles\x1b[0m nbytes          \x1b[35m%'lu\x1b[0m\n", gpus[EV_GPU_MAIN]->ntriangles*sizeof(triangle_t));

  putchar(0x0a);
  printf("\x1b[32mnintersections\x1b[0m any frame  \x1b[94m%'lu\x1b[0m\n", nintersections);
  printf("\x1b[32mnintersections\x1b[0m all frames \x1b[35m%'lu\x1b[0m\n", nintersections * EV_NFRAMES);

  // ----------------------------------------------------------------------------------------------------------------------------#
  puts("");
  for(int frame=0; frame<EV_NFRAMES; ++frame){
    printf("\x1b[35m%04d\x1b[0m \x1b[31m%6.3f\x1b[0m  ", frame, t);  fflush(stdout);

    // ----------------------------------------------------------------
    quat rot_yz = versor(-.09*M_TAU, vec3(1,0,0));
    quat rot_zx = versor(-.03*M_TAU, vec3(0,1,0));
    quat rot_xy = versor(+.01*t,     vec3(0,0,1));
    for(int gpu=0; gpu<EV_NGPUS; ++gpu){
      gpus[gpu]->t   = t;
      gpus[gpu]->rot = rot_xy * rot_zx * rot_yz;
    }

    // ----------------------------------------------------------------
    dt_ini(&tdel);
    for(int gpu=0; gpu<EV_NGPUS; ++gpu) gpu_mesh_shader(gpus[gpu]);
    for(int gpu=0; gpu<EV_NGPUS; ++gpu) gpu_sync(gpus[gpu]);  // No need to sync here!
    dt_end(&tdel);  spf=dt_del(&tdel);  printf("mesh_shdr \x1b[32m%.6f\x1b[0m  ", spf);  fflush(stdout);

    // ----------------------------------------------------------------
    dt_ini(&tdel);
    for(int gpu=0; gpu<EV_NGPUS; ++gpu) gpu_pixel_shader(gpus[gpu], gpus[EV_GPU_MAIN]->img_full_cpu, seed);
    for(int gpu=0; gpu<EV_NGPUS; ++gpu) gpu_sync(gpus[gpu]);
    dt_end(&tdel);  spf=dt_del(&tdel);  printf("px_shdr \x1b[32m%.3f\x1b[0m  px/s \x1b[94m%'.0f\x1b[0m  ", spf, (gpus[EV_GPU_MAIN]->img_w*gpus[EV_GPU_MAIN]->img_h)/spf);  fflush(stdout);
    printf("%s \x1b[31m%'.0f\x1b[0m  ", "prim/s",  (f64)gpus[EV_GPU_MAIN]->ntriangles/spf);                  fflush(stdout);
    printf("%s \x1b[32m%'.0f\x1b[0m  ", "rays/s", ((f64)nintersections/gpus[EV_GPU_MAIN]->ntriangles)/spf);  fflush(stdout);
    printf("%s \x1b[94m%'.0f\x1b[0m  ", "ints/s",  (f64)nintersections/spf);                                 fflush(stdout);

    // ----------------------------------------------------------------
    dt_ini(&tdel);  // 1920x1024 @ 80ms!
    gpu_img_save(gpus[EV_GPU_MAIN], EV_IMG_DIR, frame);
    dt_end(&tdel);  spf=dt_del(&tdel);  printf("%s \x1b[32m%.3f\x1b[0m  ", "ppm", spf);  fflush(stdout);

    // ----------------------------------------------------------------
    putchar(0x0a);
    t += dt;
  }

  // ----------------------------------------------------------------------------------------------------------------------------#
  puts(""); dt_ini(&tdel);
  for(int gpu=0; gpu<EV_NGPUS; ++gpu) gpu_work_free(gpus[gpu]);
  dt_end(&tdel);  printf("%s \x1b[33m%.6f\x1b[0m\n", "gpus_free", dt_del(&tdel));
}

// ----------------------------------------------------------------------------------------------------------------------------#
int main(){
  setlocale(LC_NUMERIC, "");
  gpu_work_t* gpus[EV_NGPUS];
  u32 img_w_min,img_w_max, img_h_min,img_h_max, img_tile_stride;

  dt_t tdel; dt_ini(&tdel);
  img_w_min       = 0;
  img_w_max       = EV_IMG_W;
  img_h_min       = 0*EV_IMG_H/EV_NGPUS;
  img_h_max       = 1*EV_IMG_H/EV_NGPUS + 64;
  img_tile_stride = img_h_min*EV_IMG_W;  // This is for the final copy of the rendered tiles from all GPUs to CPU memory!
  gpus[0]         = gpu_work_init(0, EV_IMG_W,EV_IMG_H, img_w_min,img_w_max,img_h_min,img_h_max, img_tile_stride);

  img_w_min       = 0;
  img_w_max       = EV_IMG_W;
  img_h_min       = 1*EV_IMG_H/EV_NGPUS + 64;
  img_h_max       = 2*EV_IMG_H/EV_NGPUS;
  img_tile_stride = img_h_min*EV_IMG_W;  // This is for the final copy of the rendered tiles from all GPUs to CPU memory!
  gpus[1]         = gpu_work_init(1, EV_IMG_W,EV_IMG_H, img_w_min,img_w_max,img_h_min,img_h_max, img_tile_stride);
  dt_end(&tdel);  printf("%s \x1b[33m%.6f\x1b[0m\n", "gpus_init", dt_del(&tdel));

  gpus_render_to_disk(gpus);
}

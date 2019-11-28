// nvcc evert-cli.cu -o evert-cli  -use_fast_math  &&  ./evert-cli
#include <stdint.h>
#include <stdio.h>
#define M_TAU  6.283185

// PIXEL SHADER data!
#define EV_NFRAMES  3

// MESH shader data! @theta is the AZIMUTHAL parameter; @v is the POLAR parameter!
#define EV_EPSILON       0.001f
#define EV_NSTRIPS       8
#define EV_THETA_MIN     (0)
#define EV_PHI_MIN       (0 + EV_EPSILON)
#define EV_THETA_MAX     ((8./EV_NSTRIPS)*M_TAU)  // 8
#define EV_PHI_MAX       ((2./2)         *M_PI)   // 2
#define EV_THETA_NVERTS  (1* 8*(EV_THETA_MAX-EV_THETA_MIN)/M_TAU*EV_NSTRIPS)
#define EV_PHI_NVERTS    (1*12*(EV_PHI_MAX  -EV_PHI_MIN)  /M_PI *2)
#define EV_NLIGHTS       7
#define EV_RGB_FRONT     0xff6666
#define EV_RGB_BACK      0x1188ff

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

typedef float     f32;
typedef uint32_t  u32;

// ----------------------------------------------------------------------------------------------------------------------------#
struct vec3{  // Just a simple 3D vector!
  union{  // Access the `vec3` using array notation of by specifying the name of a component!
    f32 data[3];
    struct{ f32 x0, x1, x2; };
  };
  __device__ __host__ vec3(){}
  __device__ __host__ vec3(f32 a0, f32 a1, f32 a2){  this->x0=a0; this->x1=a1; this->x2=a2;  }
  __device__ __host__ f32 operator[](int idx){       return this->data[idx];  }
};
__device__ __host__ vec3 operator*(f32   s, vec3  v){  return vec3(s*v[0], s*v[1], s*v[2]);  }
__device__ __host__ vec3 operator+(vec3 v0, vec3 v1){  return vec3(v0[0]+v1[0], v0[1]+v1[1], v0[2]+v1[2]);  }
__device__ __host__ vec3 operator-(vec3 v0, vec3 v1){  return vec3(v0[0]-v1[0], v0[1]-v1[1], v0[2]-v1[2]);  }
__device__ __host__ vec3 operator*(vec3 v0, vec3 v1){  return vec3(v0[0]*v1[0], v0[1]*v1[1], v0[2]*v1[2]);  }

// ----------------------------------------------------------------------------------------------------------------------------#
__forceinline__ __device__ vec3 clamp01(vec3 v){  return {__saturatef(v[0]), __saturatef(v[1]), __saturatef(v[2])};  }
__forceinline__ __device__ vec3 bgr8u_to_rgbf32(u32 bgr8u){
  return vec3(((bgr8u>>0x10) & 0xff)/255.,
              ((bgr8u>>0x08) & 0xff)/255.,
              ((bgr8u>>0x00) & 0xff)/255.);
}
__forceinline__ __device__ u32 rgbf32_to_bgr8u(vec3 rgbf32){
  return ((u32)(255.*rgbf32[0] + .5) << 0x10) |
         ((u32)(255.*rgbf32[1] + .5) << 0x08) |
         ((u32)(255.*rgbf32[2] + .5) << 0x00);
}

// ----------------------------------------------------------------------------------------------------------------------------#
// @section Geometric data structures! Each geometric primitive needs its own intersection routine!
struct triangle_t{
  // Intersection data!
  vec3 vert0;   // Geometry: main vertex!
  vec3 edge01;  // Geometry: vert1 - vert0
  vec3 edge02;  // Geometry: vert2 - vert0

  // Rendering data!
  u32 albedo_front;  // Lighting: albedo! Albedo is the base color input, commonly known as a diffuse map.
  u32 albedo_back;   // Lighting: albedo! Albedo is the base color input, commonly known as a diffuse map.
};




// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// @block evert code, by Nathaniel Thurston!

// @section A 1-jet, aka. a first-order jet, aka. a scalar field (evaluated at some point) together with its 1st-order partial derivatives (evaluated at some point)!
struct jet{
  f32 f;       // Scalar value of a 2D scalar field!
  f32 fu, fv;  // 1st-order partial derivatives of a 2D scalar field!
  __forceinline__ __device__ jet(){}
  __forceinline__ __device__ jet(f32 s){                  f=s; fu=0;  fv=0;   }
  __forceinline__ __device__ jet(f32 s, f32 su, f32 sv){  f=s; fu=su; fv=sv;  }
};
__forceinline__ __device__ jet operator-(jet x){           return {-x.f, -x.fu, -x.fv};  }  // Unary negation!
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

__forceinline__ __device__ jet ev_interpolate( jet x0, jet x1, jet t){  return (jet(1)-t)*x0 + t*x1;  }
__forceinline__ __device__ jet ev_partial_diff(jet  x, int idx){        return {idx==0 ? x.fu : x.fv, 0,0};  }  // Keep the partial WRT u, or the partial WRT v? It's a bug to pass a derivative index other than 0 or 1

__forceinline__ __device__ jet ev_cos(jet x){  f32 c=cosf(x.f);  f32 dc=-sinf(x.f);  return {c, dc*x.fu, dc*x.fv};  }  // Derivatives of the cosine of a scalar field!
__forceinline__ __device__ jet ev_sin(jet x){  f32 s=sinf(x.f);  f32 ds= cosf(x.f);  return {s, ds*x.fu, ds*x.fv};  }  // Derivatives of the sine of a scalar field!

// ----------------------------------------------------------------------------------------------------------------------------#
// @section A 3D vector of 1-jets!
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
// @section A quaternion of 1-jets!
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
__forceinline__ __device__ qjet ev_conj(qjet q){  return {q.x0, -q.x1, -q.x2, -q.x3};  }  // The quaternion inverse of a quaternion `q` is just `conj(q) / quad(q)`, just like for complex numbers!

__forceinline__ __device__ qjet ev_versor(jet angle, vjet dir){
  return {ev_cos(.5*angle), ev_sin(.5*angle)*ev_normalize(dir)};  // If @dir isn't a `direction vector` (ie. a unit vector), then the rotation speed is not constant, methinks!
}
__forceinline__ __device__ vjet ev_rot3d(vjet v, qjet versor){
  qjet p_rot  = ev_conj(versor) * qjet(0,v) * versor;  // Right-conjugation by @versor! The quaternion-conjugate of a unit-quaternion is its quaternion-inverse!
  return {p_rot.x1, p_rot.x2, p_rot.x3};
}
__forceinline__ __device__ vjet ev_rotx(vjet v, jet angle){  return ev_rot3d(v, ev_versor(angle, {jet(1),jet(0),jet(0)})); }
__forceinline__ __device__ vjet ev_roty(vjet v, jet angle){  return ev_rot3d(v, ev_versor(angle, {jet(0),jet(1),jet(0)})); }
__forceinline__ __device__ vjet ev_rotz(vjet v, jet angle){  return ev_rot3d(v, ev_versor(angle, {jet(0),jet(0),jet(1)})); }

// ----------------------------------------------------------------------------------------------------------------------------#
// @section geometric deformations!
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
  return -16/(M_PI*M_PI*M_PI)*(phi^3) + 12/(M_PI*M_PI)*(phi^2);  // $\purple-{16 \over \red\pi^3} \purple\dot \blue\varphi^3 ~\purple+~ {12 \over \red\pi^2} \purple\dot \blue\varphi^2$
}


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// @section eversion stages! A 3-dimensional vjet (of 1-jets, 2-jets, or 3-jets, or k-jets, or even of a mixture of jets of various degrees) always represents a point in R3! So, if the output of a function is a vjet, then we can always render its output as a vertex in R3, to visualize what in Equestria it does!

// ----------------------------------------------------------------------------------------------------------------------------#
// low-level stages!
__forceinline__ __device__ vjet ev_stage1(jet phi){  return ev_sphere_arc(phi,+1,+1,+1);  }
__forceinline__ __device__ vjet ev_stage2(jet phi){  return ev_interpolate(ev_sphere_arc(ev_phi_deform0(phi),+.9,+.9,-1), ev_sphere_arc(ev_phi_deform1(phi),+1,+1,+.5), ev_phi_deform2(phi));  }
__forceinline__ __device__ vjet ev_stage3(jet phi){  return ev_interpolate(ev_sphere_arc(ev_phi_deform0(phi),-.9,-.9,-1), ev_sphere_arc(ev_phi_deform1(phi),-1,+1,-.5), ev_phi_deform2(phi));  }
__forceinline__ __device__ vjet ev_stage4(jet phi){  return ev_sphere_arc(phi,-1,-1,-1);  }

// ----------------------------------------------------------------------------------------------------------------------------#
// mid-level stages!
__forceinline__ __device__ vjet ev_scene12(jet phi, f32 t){  return ev_interpolate(ev_stage1(phi), ev_stage2(phi), jet(t,0,0));  }
__forceinline__ __device__ vjet ev_scene23(jet phi, f32 t){  // The heart of the TWIST stage! Notice the rotations here! =D
  t *= .5;
  f32  tt    = (phi.f<=M_PI/2) ? t : -t;
  vjet rot_z = ev_rotz(ev_sphere_arc(ev_phi_deform0(phi),+0.9,+0.9,-1.0), M_TAU*jet(tt,0,0));
  vjet rot_y = ev_roty(ev_sphere_arc(ev_phi_deform1(phi),+1.0,+1.0,+0.5), M_TAU*jet(t, 0,0));
  return ev_interpolate(rot_z, rot_y, ev_phi_deform2(phi));
}
__forceinline__ __device__ vjet ev_scene34(jet phi, f32 t){  return ev_interpolate(ev_stage3(phi), ev_stage4(phi), jet(t,0,0));  }

// ----------------------------------------------------------------------------------------------------------------------------#
// high-level stages!
__forceinline__ __device__ vjet ev_figure8(vjet w, vjet h, vjet bend, jet form, jet theta){  // At the end of the twisting phase, the corrugations have nearly become figure eights!
  theta = theta%1;
  jet height = 1 - ev_cos(2*M_TAU*theta);
  if(.25<theta.f && theta.f<.75) height = 4-height;
  height = .6*height;
  h      = h + (height*height)/(8*8) * bend;
  form   = 2*form - form*form;
  return ev_sin(2*M_TAU*theta)*w + ev_interpolate(2-2*ev_cos(M_TAU*theta), height, form)*h;
}

__forceinline__ __device__ vjet ev_add_figure8(vjet p, jet theta, jet phi, jet form){
  jet size = -0.2 * ev_phi_deform2(phi) * form;  // 0.2 is like a scale constant?

  vjet du = ev_normalize(ev_partial_diff(p,0));  // Is this the partial with respect to theta, or with respect to phi?
  vjet dv = ev_normalize(ev_partial_diff(p,1));  // Is this the partial with respect to theta, or with respect to phi?
  vjet h  = 1.0*size * ev_normalize(ev_cross(du,dv));
  vjet w  = 1.1*size * ev_normalize(ev_cross(h, du));  // The 1.1 factor gives more thickness/width to the corrugations?

  vjet bend = ev_partial_diff(size,0)/ev_partial_diff(phi,0) * du;
  vjet fig8 = ev_figure8(w,h, bend, form, (f32)EV_NSTRIPS/(f32)M_TAU*theta);
  return ev_rotz(p+fig8, theta);
}


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// @section eversion phases!
__device__ vjet ev_corrugate(  jet theta, jet phi, f32 t){  vjet p=ev_stage1( phi  );  return ev_add_figure8(p, theta,phi, jet(t)  *ev_phi_deform2(phi));  }
__device__ vjet ev_push(       jet theta, jet phi, f32 t){  vjet p=ev_scene12(phi,t);  return ev_add_figure8(p, theta,phi, jet(1)  *ev_phi_deform2(phi));  }
__device__ vjet ev_twist(      jet theta, jet phi, f32 t){  vjet p=ev_scene23(phi,t);  return ev_add_figure8(p, theta,phi, jet(1)  *ev_phi_deform2(phi));  }
__device__ vjet ev_unpush(     jet theta, jet phi, f32 t){  vjet p=ev_scene34(phi,t);  return ev_add_figure8(p, theta,phi, jet(1)  *ev_phi_deform2(phi));  }
__device__ vjet ev_uncorrugate(jet theta, jet phi, f32 t){  vjet p=ev_stage4( phi  );  return ev_add_figure8(p, theta,phi, jet(1-t)*ev_phi_deform2(phi));  }




// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// @section This is the "mesh driver", ie. it takes to Nathaniel Thurton's jet/evert stuff and actually creates the sphere eversion animation! It creates vertex coordinates OUT OF THIN AIR (ie. out of kernel coordinates), per FRAME! How sexy is that? 0.5M triangles in 0.5ms!
__global__ void ker_mesh_shader(f32 t, u32 theta_nverts, u32 phi_nverts, triangle_t* triangles){
  u32 x       = blockIdx.x*blockDim.x + threadIdx.x;
  u32 y       = blockIdx.y*blockDim.y + threadIdx.y;
  u32 thr_idx = (blockIdx.y*gridDim.x + blockIdx.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x;  // Global thread index, see richiesams blogspot!

  f32 theta_range = EV_THETA_MAX - EV_THETA_MIN;
  f32 phi_range   = EV_PHI_MAX   - EV_PHI_MIN;
  f32 theta       = EV_THETA_MIN + (f32)x/theta_nverts * theta_range;     // Now is theta in [0 .. EV_THETA_MAX)
  f32 phi         = EV_PHI_MIN   + (f32)y/phi_nverts   * phi_range;       // Now is phi   in (0 .. EV_PHI_MAX)
  f32 dtheta      = 1 / (theta_nverts + .05*theta_nverts) * theta_range;  // 0.5 0.5  .5  .50 1.0  8 4 2 1 2
  f32 dphi        = 1 / (phi_nverts   + .05*phi_nverts)   * phi_range;    // 1.0 2.0  .5  .25 0.5  1 1 1 1 2

  vjet vert0_jet, vert1_jet, vert2_jet, vert3_jet;  // MAIN geometry driver!
  if(t-EV_EPSILON < EV_CORRUGATE_TINI+EV_CORRUGATE_TDEL){  // WARN! For some reason we need to subtract the time by EV_EPSILON?
    vert0_jet = ev_corrugate(jet(theta + 0*dtheta), jet(phi + 0*dphi, 1,0), (t-EV_CORRUGATE_TINI)/EV_CORRUGATE_TDEL);
    vert1_jet = ev_corrugate(jet(theta + 0*dtheta), jet(phi + 1*dphi, 1,0), (t-EV_CORRUGATE_TINI)/EV_CORRUGATE_TDEL);
    vert2_jet = ev_corrugate(jet(theta + 1*dtheta), jet(phi + 0*dphi, 1,0), (t-EV_CORRUGATE_TINI)/EV_CORRUGATE_TDEL);
    vert3_jet = ev_corrugate(jet(theta + 1*dtheta), jet(phi + 1*dphi, 1,0), (t-EV_CORRUGATE_TINI)/EV_CORRUGATE_TDEL);
  }else if(t-EV_EPSILON < EV_PUSH_TINI+EV_PUSH_TDEL){
    vert0_jet = ev_push(jet(theta + 0*dtheta), jet(phi + 0*dphi, 1,0), (t-EV_PUSH_TINI)/EV_PUSH_TDEL);
    vert1_jet = ev_push(jet(theta + 0*dtheta), jet(phi + 1*dphi, 1,0), (t-EV_PUSH_TINI)/EV_PUSH_TDEL);
    vert2_jet = ev_push(jet(theta + 1*dtheta), jet(phi + 0*dphi, 1,0), (t-EV_PUSH_TINI)/EV_PUSH_TDEL);
    vert3_jet = ev_push(jet(theta + 1*dtheta), jet(phi + 1*dphi, 1,0), (t-EV_PUSH_TINI)/EV_PUSH_TDEL);
  }else if(t-EV_EPSILON < EV_TWIST_TINI+EV_TWIST_TDEL){
    vert0_jet = ev_twist(jet(theta + 0*dtheta), jet(phi + 0*dphi, 1,0), (t-EV_TWIST_TINI)/EV_TWIST_TDEL);
    vert1_jet = ev_twist(jet(theta + 0*dtheta), jet(phi + 1*dphi, 1,0), (t-EV_TWIST_TINI)/EV_TWIST_TDEL);
    vert2_jet = ev_twist(jet(theta + 1*dtheta), jet(phi + 0*dphi, 1,0), (t-EV_TWIST_TINI)/EV_TWIST_TDEL);
    vert3_jet = ev_twist(jet(theta + 1*dtheta), jet(phi + 1*dphi, 1,0), (t-EV_TWIST_TINI)/EV_TWIST_TDEL);
  }else if(t-EV_EPSILON < EV_UNPUSH_TINI+EV_UNPUSH_TDEL){
    vert0_jet = ev_unpush(jet(theta + 0*dtheta), jet(phi + 0*dphi, 1,0), (t-EV_UNPUSH_TINI)/EV_UNPUSH_TDEL);
    vert1_jet = ev_unpush(jet(theta + 0*dtheta), jet(phi + 1*dphi, 1,0), (t-EV_UNPUSH_TINI)/EV_UNPUSH_TDEL);
    vert2_jet = ev_unpush(jet(theta + 1*dtheta), jet(phi + 0*dphi, 1,0), (t-EV_UNPUSH_TINI)/EV_UNPUSH_TDEL);
    vert3_jet = ev_unpush(jet(theta + 1*dtheta), jet(phi + 1*dphi, 1,0), (t-EV_UNPUSH_TINI)/EV_UNPUSH_TDEL);
  }else if(t-EV_EPSILON < EV_UNCORRUGATE_TINI+EV_UNCORRUGATE_TDEL){
    vert0_jet = ev_uncorrugate(jet(theta + 0*dtheta), jet(phi + 0*dphi, 1,0), (t-EV_UNCORRUGATE_TINI)/EV_UNCORRUGATE_TDEL);
    vert1_jet = ev_uncorrugate(jet(theta + 0*dtheta), jet(phi + 1*dphi, 1,0), (t-EV_UNCORRUGATE_TINI)/EV_UNCORRUGATE_TDEL);
    vert2_jet = ev_uncorrugate(jet(theta + 1*dtheta), jet(phi + 0*dphi, 1,0), (t-EV_UNCORRUGATE_TINI)/EV_UNCORRUGATE_TDEL);
    vert3_jet = ev_uncorrugate(jet(theta + 1*dtheta), jet(phi + 1*dphi, 1,0), (t-EV_UNCORRUGATE_TINI)/EV_UNCORRUGATE_TDEL);
  }

  // ----------------------------------------------------------------------------------------------------------------------------#
  vec3 vert0 = vec3(vert0_jet.x0.f, vert0_jet.x1.f, vert0_jet.x2.f);
  vec3 vert1 = vec3(vert1_jet.x0.f, vert1_jet.x1.f, vert1_jet.x2.f);
  vec3 vert2 = vec3(vert2_jet.x0.f, vert2_jet.x1.f, vert2_jet.x2.f);
  vec3 vert3 = vec3(vert3_jet.x0.f, vert3_jet.x1.f, vert3_jet.x2.f);

  vec3 color0  = bgr8u_to_rgbf32(EV_RGB_FRONT);  // sin(theta): as `theta` goes from 0 to TAU, `sin(theta)` goes from 0 to 0
  vec3 color1  = bgr8u_to_rgbf32(EV_RGB_BACK);   // sin(2*phi): as `phi`   goes from 0 to PI,  `sin(2*phi)` goes from 0 to 0
  vec3 dcolor0 = .15 * vec3(0,0,(sin(theta)+1)/2);
  vec3 dcolor1 = .30 * vec3((sin(theta)+1)/2,0,0);

  triangle_t triangle;
  triangle.albedo_front = rgbf32_to_bgr8u(clamp01(color0 + dcolor0));
  triangle.albedo_back  = rgbf32_to_bgr8u(clamp01(color1 + dcolor1));
  triangle.vert0=vert0;  triangle.edge01=vert1-vert0;  triangle.edge02=vert2-vert0;  triangles[2*thr_idx+0]=triangle;
  triangle.vert0=vert3;  triangle.edge01=vert2-vert3;  triangle.edge02=vert1-vert3;  triangles[2*thr_idx+1]=triangle;
}




// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
int main(){
  cudaSetDevice(0);
  u32  theta_nverts   = ceilf(EV_THETA_NVERTS);
  u32  phi_nverts     = ceilf(EV_PHI_NVERTS);
  f32  t              = EV_TMIN;
  f32  dt             = (EV_TMAX-EV_TMIN) / (EV_NFRAMES-1);
  dim3 MESH_BLOCK_DIM = {1,1,1};  // {1,1,1} {8,8,1} {32,32,1}                            // Launch `MESH_BLOCK_DIM.x * MESH_BLOCK_DIM.y * MESH_BLOCK_DIM.z` nthreads per block! So, `32 * 32 * 1` nthreads per block! Max nthreads per block on Titan V is 1024!
  dim3 MESH_GRID_DIM  = {theta_nverts/MESH_BLOCK_DIM.x, phi_nverts/MESH_BLOCK_DIM.y, 1};  // Launch ` MESH_GRID_DIM.x *  MESH_GRID_DIM.y *  MESH_GRID_DIM.z` nblocks  per grid!  Then `nthreads per grid` is `nblocks per grid  *  nthreads per block`!
  triangle_t* triangles_gpu; cudaMalloc(    &triangles_gpu, sizeof(triangle_t)*theta_nverts*phi_nverts*2);
  triangle_t* triangles_cpu; cudaMallocHost(&triangles_cpu, sizeof(triangle_t)*theta_nverts*phi_nverts*2);

  // ----------------------------------------------------------------
  printf("nframes      \x1b[94m%d\x1b[0m\n",    EV_NFRAMES);
  printf("ntriangles   \x1b[94m%'u\x1b[0m\n",   theta_nverts*phi_nverts*2);
  printf("theta nverts \x1b[94m%'u\x1b[0m\n",   theta_nverts);
  printf("phi   nverts \x1b[94m%'u\x1b[0m\n",   phi_nverts);
  printf("mesh grid    \x1b[94m%d %d\x1b[0m\n", MESH_GRID_DIM.x, MESH_GRID_DIM.y);

  // ----------------------------------------------------------------
  for(int frame=0; frame<EV_NFRAMES; ++frame){
    ker_mesh_shader<<<MESH_GRID_DIM,MESH_BLOCK_DIM>>>(t,theta_nverts,phi_nverts, triangles_gpu);
    cudaMemcpy(triangles_cpu, triangles_gpu, sizeof(triangle_t)*theta_nverts*phi_nverts*2, cudaMemcpyDeviceToHost);
    t += dt;

    printf("\nframe:\x1b[91m%d\x1b[0m\n", frame);
    for(int i=0; i<theta_nverts*phi_nverts*2; ++i)
      printf("triangle:\x1b[35m%4d\x1b[0m  v0:\x1b[31m%6.3f\x1b[0m \x1b[32m%6.3f\x1b[0m \x1b[94m%6.3f\x1b[0m  e01:\x1b[31m%6.3f\x1b[0m \x1b[32m%6.3f\x1b[0m \x1b[94m%6.3f\x1b[0m  e02:\x1b[31m%6.3f\x1b[0m \x1b[32m%6.3f\x1b[0m \x1b[94m%6.3f\x1b[0m  f:\x1b[35m%06x\x1b[0m b:\x1b[94m%06x\x1b[0m\n",
        i, triangles_cpu[i].vert0[0],triangles_cpu[i].vert0[1],triangles_cpu[i].vert0[2], triangles_cpu[i].edge01[0],triangles_cpu[i].edge01[1],triangles_cpu[i].edge01[2], triangles_cpu[i].edge02[0],triangles_cpu[i].edge02[1],triangles_cpu[i].edge02[2], triangles_cpu[i].albedo_front, triangles_cpu[i].albedo_back);
  }

  // ----------------------------------------------------------------
  cudaFree(triangles_gpu);
  cudaFreeHost(triangles_cpu);
  cudaDeviceReset();
}

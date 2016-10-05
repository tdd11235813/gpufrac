#ifndef __CUDA_GLOBALS_H
#define __CUDA_GLOBALS_H

#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
// ----------------------------------------------------------------------------
template<typename T>
struct Data
{
  T* buffer = nullptr;
};

enum class Fractal {
    POPCORN0=0,
    POPCORN1, POPCORN2, POPCORN3, MCCABE,
    _COUNT
};

/// Parameters
template<typename T>
struct Parameters
{
  T t0 = 3.11;
  T t1 = -4.34;
  T t2 = -4.33;
  T t3 = 2.22;
  T x0=-6.0;
  T x1= 6.0;
  T y0=-6.0;
  T y1= 6.0;
  T talpha = 0.1;
  T addValue = 0.01;
  T time = 0.0; // elapsed
  T time_delta = 0.0; // time delta to last frame
  unsigned width=800;
  unsigned height=800;
  unsigned max_iterations=64;
  unsigned iterations=64;
  unsigned iterations_per_run=32;
  unsigned n=0;
  bool use_atomics = true;
  bool invert = false;
  bool sub_sampling = false;
  T hue_start = 0.0;
  T hue_end   = 0.3;
  T hue_slope = 1.45;
  T density_slope = 1.0;
  T brightness_slope = 1.0;
  T saturation_slope = 1.0;

};
/// User Setting
struct UserSetting
{
  bool hslMode=false;
  Fractal fractal=Fractal::POPCORN0;
  bool animation=false;
  bool vsync = false;
  double timeScale = 0.15;

  std::string outputDir="output";
  std::string prefix="img_";
};

template<unsigned FuncId, bool HSB, typename T>
float launch_kernel(
    cudaGraphicsResource* dst,
    Data<T>& ddata,
    const Parameters<T>& params,
    unsigned);

template<typename T>
void alloc_buffer(
    Data<T>& ddata,
    const Parameters<T>& params);

template<typename T>
void init_buffer(
    Data<T>& ddata,
    const Parameters<T>& params);

template<typename T>
void cleanup_cuda(Data<T>& ddata);

// ----------------------------------------------------------------------------

///
template<typename T>
struct DataMc
{
  T* buffer = nullptr;
  int levels = -1;
  int blurlevels = 1;
  unsigned *radii = nullptr;
  unsigned *radii_host = nullptr;
  T *stepSizes = nullptr;
  T *colorShift = nullptr;

  int* bestLevel = nullptr;
  bool* direction = nullptr; /*@todo check for other type*/

  //T scale; // space zoom (scale)
  T base = 2.0;
  T stepScale = 0.002;
  T stepOffset = 0.002;
  float blurFactor = 1.0f;
  int symmetry = 0;

  T* SATs[2] = {nullptr, nullptr};
};

template<typename T>
void init_buffer(DataMc<T>&,
                 const Parameters<T>& parameters,
                 bool alloc);
template<typename T>
float launch_kernel(cudaGraphicsResource* dst,
                   DataMc<T>& ddata,
                   const Parameters<T>& params);

template<typename T>
void cleanup_cuda(DataMc<T>& ddata);
// ----------------------------------------------------------------------------


#ifndef CUDA_DISABLE_ERROR_CHECKING
#define CHECK_CUDA(ans) check_cuda((ans), "", #ans, __FILE__, __LINE__)
#define CHECK_LAST(msg) check_cuda(cudaGetLastError(), msg, "CHECK_LAST", __FILE__, __LINE__)
#else
#define CHECK_CUDA(ans) {}
#define CHECK_LAST(msg) {}
#endif


inline
void throw_error(int code,
                 const char* error_string,
                 const char* msg,
                 const char* func,
                 const char* file,
                 int line) {
  throw std::runtime_error("CUDA error "
                           +std::string(msg)
                           +" "+std::string(error_string)
                           +" ["+std::to_string(code)+"]"
                           +" "+std::string(file)
                           +":"+std::to_string(line)
                           +" "+std::string(func)
    );
}

inline
void check_cuda(cudaError_t code, const char* msg, const char *func, const char *file, int line) {
  if (code != cudaSuccess) {
    throw_error(static_cast<int>(code),
                cudaGetErrorString(code), msg, func, file, line);
  }
}


inline
std::stringstream getCUDADeviceInformations(int dev) {
  std::stringstream info;
  cudaDeviceProp prop;
  int runtimeVersion = 0;
  size_t f=0, t=0;
  CHECK_CUDA( cudaRuntimeGetVersion(&runtimeVersion) );
  CHECK_CUDA( cudaGetDeviceProperties(&prop, dev) );
  CHECK_CUDA( cudaMemGetInfo(&f, &t) );
  info << '"' << prop.name << '"'
       << ", \"CC\", " << prop.major << '.' << prop.minor
       << ", \"Multiprocessors\", "<< prop.multiProcessorCount
       << ", \"Memory [MiB]\", "<< t/1048576
       << ", \"MemoryFree [MiB]\", " << f/1048576
       << ", \"MemClock [MHz]\", " << prop.memoryClockRate/1000
       << ", \"GPUClock [MHz]\", " << prop.clockRate/1000
       << ", \"CUDA Runtime\", " << runtimeVersion
    ;
  return info;
}

inline
std::stringstream listCudaDevices() {
  std::stringstream info;
  int nrdev = 0;
  CHECK_CUDA( cudaGetDeviceCount( &nrdev ) );
  if(nrdev==0)
    throw std::runtime_error("No CUDA capable device found");
  for(int i=0; i<nrdev; ++i)
    info << "\"ID\"," << i << "," << getCUDADeviceInformations(i).str() << std::endl;
  return info;
}

#endif

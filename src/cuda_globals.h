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
    POPCORN1, POPCORN2, POPCORN3,
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
  T time = 0.0;
  unsigned width=800;
  unsigned height=800;
  unsigned n=0;
  unsigned max_iterations=64;
  T hueOffset=0.0;
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
    const Parameters<T>& params);

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

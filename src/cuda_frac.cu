#include "cuda_globals.h"

constexpr static double PI = 3.141592653589793;
static cudaEvent_t custart, cuend;
// ---

template<typename T>
__device__
unsigned unmap( T v, const T v0, const T v1, const T len)
{
  return static_cast<unsigned>( (v-v0)/(v1-v0)*len );
}

template<typename T>
__device__
T map( unsigned v, const T v0, const T v1, const T len)
{
  return static_cast<T>(v)/len*(v1-v0)+v0;
}


__device__ inline
unsigned char toColor(float v) {
  return static_cast<unsigned char>(255.0f*saturate(v));
}


/// HSL [0:1] to RGB {0..255}, from http://stackoverflow.com/questions/4728581/hsl-image-adjustements-on-gpu
__device__
void hsl2rgb( float hue, float sat, float lum, uchar4& color )
{
    const float onethird = 1.0 / 3.0;
    const float twothird = 2.0 / 3.0;
    const float rcpsixth = 6.0;


    float xtr =    rcpsixth * (hue - twothird);
    float xtg =   0.0;
    float xtb =    rcpsixth * (1.0 - hue);

    if (hue < twothird) {
        xtr = 0.0;
        xtg = rcpsixth * (twothird - hue);
        xtb = rcpsixth * (hue      - onethird);
    }

    if (hue < onethird) {
        xtr = rcpsixth * (onethird - hue);
        xtg = rcpsixth * hue;
        xtb = 0.0;
    }

    xtr = __saturatef(xtr);
    xtg = __saturatef(xtg);
    xtb = __saturatef(xtb);

    float sat2   =  2.0 * sat;
    float satinv =  1.0 - sat;
    float luminv =  1.0 - lum;
    float lum2m1 = (2.0 * lum) - 1.0;
    float ctr    = (sat2 * xtr) + satinv;
    float ctg    = (sat2 * xtg) + satinv;
    float ctb    = (sat2 * xtb) + satinv;

    if (lum >= 0.5) {
      color.x = toColor((luminv * ctr) + lum2m1);
      color.y = toColor((luminv * ctg) + lum2m1);
      color.z = toColor((luminv * ctb) + lum2m1);
    }else {
      color.x = toColor(lum * ctr);
      color.y = toColor(lum * ctg);
      color.z = toColor(lum * ctb);
    }
}

/**
 *
 */
__global__ void d_clear_color(uchar4 *ptr, unsigned n)
{
  unsigned i=threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n)
    return;
  ptr[i].x = 0;
  ptr[i].y = 0;
  ptr[i].z = 0;
}

template<unsigned FuncId, typename T>
inline __device__
T funcX(T t, const T time, T xk, T yk, const Parameters<T>& params) {
  switch(FuncId) {
    case 0: return xk+params.talpha*cos( params.t0+time+yk+cos(params.t1+time+PI*xk));
    case 1: return t*yk+0.95f*xk-params.talpha*sin(0.7f*yk);//+sin(3.0f*0.7f*yk));
    // xk-hf(y+hf(x)), f(x)=sin(x+sin(3x))
    case 2: return xk-params.talpha*sin( yk+params.talpha*sin(params.t0+time+xk+sin(params.t1+time+3.0*xk)) + sin(3.0*(yk+params.talpha*sin(params.t0+time+xk+sin(params.t1+time+3.0*xk)))) );
    case 3: return xk-params.talpha*sin( params.t0+yk+time+sin(3*yk+params.t1+time+sin(2*yk+time)) );
  }
  return xk;
}

template<unsigned FuncId, typename T>
inline __device__
T funcY(T t, T time, T xk, T yk, const Parameters<T>& params) {
  switch(FuncId) {
    case 0: return yk+params.talpha*cos( params.t2+time+xk+cos(params.t3+time+PI*yk));
    case 1: return t*xk+0.95f*yk+params.talpha*sin(0.6f*xk);//+sin(3.0f*0.6f*xk));
    case 2: return yk+params.talpha*sin(params.t2+time+xk+sin(params.t3+time+3.0*xk));
    case 3: return yk+params.talpha*sin( xk+params.t2+time+sin(3*xk+params.t3+time+sin(2*xk+time)) );
  }
  return yk;
}

template<unsigned FuncId, bool HSB, typename T>
__global__
void generatePattern(
      Data<T> data,
      const Parameters<T> params
    )
{
  unsigned i,j;
  unsigned px, py;
  T xk,yk;
  T width = params.width;
  T height = params.height;
  T ts = 1.0f/(params.max_iterations-1);
  T t;
  for (i = blockIdx.y * blockDim.y + threadIdx.y;
       i < params.height;
       i += blockDim.y * gridDim.y)
  {
    for (j = blockIdx.x * blockDim.x + threadIdx.x;
         j < params.width;
         j += blockDim.x * gridDim.x)
    {
      xk = map(j, params.x0, params.x1, width);
      yk = map(i, params.y0, params.y1, height);

      for(t=0.0f; t<1.0f; t+=ts) {
        xk = funcX<FuncId>(t, params.time, xk, yk, params);
        yk = funcY<FuncId>(t, params.time, xk, yk, params);
        px = unmap(xk, params.x0, params.x1, width);
        py = unmap(yk, params.y0, params.y1, height);
        if (px<params.width && py<params.height) {
          unsigned offset = px+py*params.width;
          if(HSB) {
            T v = params.addValue*t;

            atomicAdd(data.buffer+offset, v); // hue
            atomicAdd(data.buffer+offset+params.n, v); // saturation
            atomicAdd(data.buffer+offset+2*params.n, v); // brightness
          }else{
            atomicAdd(data.buffer+offset, params.addValue);
          }
        }
      }
    }
  }
}

template<bool HSB, typename T>
__global__
void renderToImage(
      uchar4 *ptr,
      Data<T> data,
      const Parameters<T> params
    )
{
  unsigned j;
  for (j = blockIdx.x * blockDim.x + threadIdx.x;
       j < params.n;
       j += blockDim.x * gridDim.x)
  {
    if(HSB)
    {
      T v = data.buffer[j];
      float h = 0.5f-powf(0.3f*__saturatef(v), 0.2f)+params.hueOffset;
      if(h>1.0f)
        h = h-1.0f;
      float s = __saturatef(powf(data.buffer[j+params.n], 0.8f));
      float l = __saturatef(powf(data.buffer[j+2*params.n], 0.8f));
      hsl2rgb(h, s, l, ptr[j]);
    }else{
      T density = sqrt(data.buffer[j]);//exp(-data.buffer[j])
      ptr[j].x = toColor( 1.0-0.3*powf(density,0.4) );
      ptr[j].y = toColor( 1.0-0.5*powf(density,1.0) );
      ptr[j].z = toColor( 1.0-0.8*powf(density,1.4) );
      /*unsigned char d = 255*data.buffer[j];//exp(-data.buffer[j])
      ptr[j].x = d;
      ptr[j].y = d;
      ptr[j].z = d;*/
    }
  }
}
/**
 *
 */
template<unsigned FuncId, bool HSB, typename T>
float launch_kernel(
    cudaGraphicsResource* dst,
    Data<T>& ddata,
    const Parameters<T>& params)
{
  int numSMs;
  int devId = 0;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);

  dim3 threads( 16, 16 );
  dim3 threads1d( 128 );
  dim3 blocks( 32*numSMs );
  uchar4* pos;
  size_t num_bytes;
  cudaError_t err;
  float ms = 0.0f;

  err=cudaGraphicsMapResources(1, &dst, 0);
  if (err == cudaSuccess)
  {
    CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(
        (void**)&pos, &num_bytes, dst));

    CHECK_CUDA(cudaEventRecord(custart));
    generatePattern<FuncId, HSB><<<blocks, threads>>>(ddata, params);
    CHECK_CUDA(cudaEventRecord(cuend));

    renderToImage<HSB><<<blocks, threads1d>>>(pos, ddata, params);

    CHECK_CUDA( cudaEventSynchronize(cuend) );
    CHECK_CUDA( cudaEventElapsedTime(&ms, custart, cuend) );
  }
  CHECK_CUDA( cudaGraphicsUnmapResources(1, &dst, 0));
  return ms;
}

/**
 *
 */
template<typename T>
void alloc_buffer(
    Data<T>& ddata,
    const Parameters<T>& params)
{
  if(ddata.buffer)
  {
    CHECK_CUDA( cudaFree(ddata.buffer) );
    CHECK_CUDA( cudaEventDestroy(custart) );
    CHECK_CUDA( cudaEventDestroy(cuend) );
  }
  unsigned n = 4 * params.n;
  CHECK_CUDA( cudaMalloc(&ddata.buffer, n*sizeof(T)) );
  CHECK_CUDA( cudaEventCreate(&custart) );
  CHECK_CUDA( cudaEventCreate(&cuend) );
}
/**
 *
 */
template<typename T>
void init_buffer(
    Data<T>& ddata,
    const Parameters<T>& params)
{
  unsigned n = 4 * params.n;
  CHECK_CUDA( cudaMemset(ddata.buffer, 0.0, n*sizeof(T)));
  CHECK_CUDA( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
}
/**
 *
 */
template<typename T>
void cleanup_cuda(Data<T>& ddata)
{
  if(ddata.buffer)
  {
    CHECK_CUDA( cudaFree(ddata.buffer) );
    ddata.buffer = 0;
  }
}


template
void alloc_buffer<float>(Data<float>&, const Parameters<float>&);
template
void init_buffer<float>(Data<float>&, const Parameters<float>&);
template float launch_kernel<0, false, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&);
template float launch_kernel<1, false, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&);
template float launch_kernel<2, false, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&);
template float launch_kernel<3, false, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&);
template float launch_kernel<0, true, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&);
template float launch_kernel<1, true, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&);
template float launch_kernel<2, true, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&);
template float launch_kernel<3, true, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&);
template
void cleanup_cuda<float>(Data<float>&);

/*
template
void alloc_buffer<double>(Data<double>&, const Parameters<double>&);
template
void init_buffer<double>(Data<double>&, const Parameters<double>&);
template float launch_kernel<0, false, double>(cudaGraphicsResource*, Data<double>&, const Parameters<double>&);
template float launch_kernel<1, false, double>(cudaGraphicsResource*, Data<double>&, const Parameters<double>&);
template float launch_kernel<2, false, double>(cudaGraphicsResource*, Data<double>&, const Parameters<double>&);
template float launch_kernel<3, false, double>(cudaGraphicsResource*, Data<double>&, const Parameters<double>&);
template float launch_kernel<0, true, double>(cudaGraphicsResource*, Data<double>&, const Parameters<double>&);
template float launch_kernel<1, true, double>(cudaGraphicsResource*, Data<double>&, const Parameters<double>&);
template float launch_kernel<2, true, double>(cudaGraphicsResource*, Data<double>&, const Parameters<double>&);
template float launch_kernel<3, true, double>(cudaGraphicsResource*, Data<double>&, const Parameters<double>&);
template
void cleanup_cuda<double>(Data<double>&);
*/

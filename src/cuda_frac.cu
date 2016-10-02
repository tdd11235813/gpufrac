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


    float xtr = rcpsixth * (hue - twothird);
    float xtg = 0.0;
    float xtb = rcpsixth * (1.0 - hue);

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


template<typename T>
__global__
void d_init_buffer(
      Data<T> _data,
      const Parameters<T> _params)
{
  unsigned i,j,offset_ij;
  T width = _params.width;
  T height = _params.height;
  for (i = blockIdx.y * blockDim.y + threadIdx.y;
       i < _params.height;
       i += blockDim.y * gridDim.y)
  {
    for (j = blockIdx.x * blockDim.x + threadIdx.x;
	 j < _params.width;
	 j += blockDim.x * gridDim.x)
    {
      offset_ij = j+i*_params.width;
      _data.buffer[offset_ij]             = 0.0f;
      _data.buffer[offset_ij+1*_params.n] = 0.0f;
      _data.buffer[offset_ij+2*_params.n] = 0.0f;
      _data.buffer[offset_ij+3*_params.n] = map(j, _params.x0, _params.x1, width);
      _data.buffer[offset_ij+4*_params.n] = map(i, _params.y0, _params.y1, height);
    }
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

template<unsigned TFuncId, typename T>
inline __device__
T funcX(T t, const T time, T xk, T yk, const Parameters<T>& params) {
  switch(TFuncId) {
    case 0: return xk+params.talpha*cos( params.t0+time+yk+cos(params.t1+time+PI*xk));
    case 1: return t*yk+0.95f*xk-params.talpha*sin(0.7f*yk);//+sin(3.0f*0.7f*yk));
    // xk-hf(y+hf(x)), f(x)=sin(x+sin(3x))
    case 2: return xk-params.talpha*sin( yk+params.talpha*sin(params.t0+time+xk+sin(params.t1+time+3.0*xk)) + sin(3.0*(yk+params.talpha*sin(params.t0+time+xk+sin(params.t1+time+3.0*xk)))) );
    case 3: return xk-params.talpha*sin( params.t0+yk+time+sin(3*yk+params.t1+time+sin(2*yk+time)) );
  }
  return xk;
}

template<unsigned TFuncId, typename T>
inline __device__
T funcY(T t, const T time, T xk, T yk, const Parameters<T>& params) {
  switch(TFuncId) {
    case 0: return yk+params.talpha*cos( params.t2+time+xk+cos(params.t3+time+PI*yk));
    case 1: return t*xk+0.95f*yk+params.talpha*sin(0.6f*xk);//+sin(3.0f*0.6f*xk));
    case 2: return yk+params.talpha*sin(params.t2+time+xk+sin(params.t3+time+3.0*xk));
    case 3: return yk+params.talpha*sin( xk+params.t2+time+sin(3*xk+params.t3+time+sin(2*xk+time)) );
  }
  return yk;
}

template<typename T>
__device__ void d_draw_line(Data<T>& _data,
		       const Parameters<T>& _params,
		       unsigned x0, unsigned y0,
		       unsigned x1, unsigned y1,
		       T v)
{
  unsigned tmp;
  if( x0 > x1 ) {
    tmp = x0;
    x0 = x1;
    x1 = tmp;
  }
  if( y0 > y1 ) {
    tmp = y0;
    y0 = y1;
    y1 = tmp;
  }
  unsigned dx = x1 - x0;
  unsigned dy = y1 - y0;
  int D = 2*dy - dx;
  unsigned y = y0;
  for(unsigned x=x0; x<x1; ++x) {
    unsigned offset = x + y*_params.width;
    atomicAdd(_data.buffer+offset, v); // hue
    atomicAdd(_data.buffer+offset+_params.n, v); // saturation
    atomicAdd(_data.buffer+offset+2*_params.n, v); // brightness
    if(D >= 0) {
      y = y + 1;
      D = D - dx;
    }
    D = D + dy;
  }
}

template<unsigned TFuncId,
	   bool TColoring,
	   typename T>
__global__
void d_generate_pattern(
      Data<T> _data,
      const Parameters<T> _params,
      const T _iteration_start,
      const T _iteration_end,
      const T _iteration_step_size
    )
{
  unsigned i,j;
  unsigned pxo=0xffff, pyo=0xffff;
  unsigned px, py;
  unsigned offset_ij;
  T xk,yk;
  T width = _params.width;
  T height = _params.height;
  T t;
  for (i = blockIdx.y * blockDim.y + threadIdx.y;
       i < _params.height;
       i += blockDim.y * gridDim.y)
  {
    for (j = blockIdx.x * blockDim.x + threadIdx.x;
	 j < _params.width;
         j += blockDim.x * gridDim.x)
    {
      offset_ij = j+i*_params.width;
      xk = _data.buffer[offset_ij + 3*_params.n];
      yk = _data.buffer[offset_ij + 4*_params.n];
      pxo=0xffff;
      pyo=0xffff;
      for(t=_iteration_start; t<_iteration_end; t+=_iteration_step_size) {
	xk = funcX<TFuncId>(t, _params.time, xk, yk, _params);
	yk = funcY<TFuncId>(t, _params.time, xk, yk, _params);
	px = unmap(xk, _params.x0, _params.x1, width);
	py = unmap(yk, _params.y0, _params.y1, height);
	if (px<_params.width && py<_params.height) {
	  unsigned offset = px+py*_params.width;
	  if(TColoring) {
	    T v = _params.addValue*t;
	    if(pxo!=0xffff && pyo!=0xffff && (pxo!=px || pyo!=py) ){
	      if( (px<pxo ? (pxo-px)>2 && (pxo-px)<50 : (px-pxo)>2 && (px-pxo)<50 ) ||
		  (py<pyo ? (pyo-py)>2 && (pyo-py)<50 : (py-pyo)>2 && (py-pyo)<50 )
		  ) {
	      // draw line to previous pixel
		d_draw_line(_data, _params,
			    px,py, pxo,pyo,
			    v*rsqrtf( (px-pxo)*(px-pxo) + (py-pyo)*(py-pyo) ) );
	      }
	    }

	    atomicAdd(_data.buffer+offset, v); // hue
	    atomicAdd(_data.buffer+offset+_params.n, v); // saturation
	    atomicAdd(_data.buffer+offset+2*_params.n, v); // brightness
	    pxo = px;
	    pyo = py;

	  }else{
	    atomicAdd(_data.buffer+offset, _params.addValue);
	  }
	}
      }
      _data.buffer[offset_ij + 3*_params.n] = xk;
      _data.buffer[offset_ij + 4*_params.n] = yk;
    }
  }
}

template<bool TColoring, typename T>
__global__
void d_render_to_image(
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
    if(TColoring)
    {
      T v = data.buffer[j];
      float h = 0.5f-powf(0.3f*__saturatef(v), 0.2f)+params.hueOffset;
      if(h>1.0f)
        h = h-1.0f;
      float s = __saturatef(powf(data.buffer[j+params.n], 0.8f));
      float l = /*1.0f-*/__saturatef(powf(data.buffer[j+2*params.n], 0.8f));
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
template<unsigned TFuncId, bool TColoring, typename T>
float launch_kernel(
    cudaGraphicsResource* _dst,
    Data<T>& _ddata,
    const Parameters<T>& _params,
    unsigned _iteration_offset)
{
  int numSMs;
  int devId = 0;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);

  dim3 threads( 16, 16 );
  dim3 threads1d( 128 );
  dim3 blocks( 32*numSMs );
  size_t num_bytes;
  cudaError_t err;
  float ms = 0.0f;

  err=cudaGraphicsMapResources(1, &_dst, 0);
  if (err == cudaSuccess)
  {
    uchar4* pos;
    const T it_start = _iteration_offset/T(_params.max_iterations);
    const T it_end = it_start + _params.iterations_per_run/T(_params.max_iterations);
    const T it_step_size = 1.0/T(_params.max_iterations);

    CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(
	(void**)&pos, &num_bytes, _dst));

    CHECK_CUDA(cudaEventRecord(custart));
    if(_iteration_offset==0) {
      d_generate_pattern<TFuncId, TColoring>
	<<<blocks, threads>>>(_ddata, _params, it_start, it_end, it_step_size);
    }else{
       d_generate_pattern<TFuncId, TColoring>
	<<<blocks, threads>>>(_ddata, _params, it_start, it_end, it_step_size);
    }

    CHECK_CUDA(cudaEventRecord(cuend));

    d_render_to_image<TColoring><<<blocks, threads1d>>>(pos, _ddata, _params);

    CHECK_CUDA( cudaEventSynchronize(cuend) );
    CHECK_CUDA( cudaEventElapsedTime(&ms, custart, cuend) );
  }
  CHECK_CUDA( cudaGraphicsUnmapResources(1, &_dst, 0));
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
  unsigned n = 5 * params.n;
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
  //unsigned n = 5 * params.n;
  //  CHECK_CUDA( cudaMemset(ddata.buffer, 0.0, n*sizeof(T)));
  int numSMs;
  int devId = 0;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);

  dim3 threads( 16, 16 );
  dim3 blocks( 32*numSMs );
  d_init_buffer<<<blocks, threads>>>(ddata, params);
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
template float launch_kernel<0, false, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&, unsigned);
template float launch_kernel<1, false, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&, unsigned);
template float launch_kernel<2, false, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&, unsigned);
template float launch_kernel<3, false, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&, unsigned);
template float launch_kernel<0, true, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&, unsigned);
template float launch_kernel<1, true, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&, unsigned);
template float launch_kernel<2, true, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&, unsigned);
template float launch_kernel<3, true, float>(cudaGraphicsResource*, Data<float>&, const Parameters<float>&, unsigned);
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

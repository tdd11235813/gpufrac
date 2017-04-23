#include "fractal_popcorn.cuh"
#include "cuda_globals.hpp"


#include <iostream>

constexpr static double PI = 3.141592653589793;
static cudaEvent_t custart, cuend;


// ---


namespace Fractal {
  namespace Popcorn {


template<typename T>
__device__
unsigned unmap( T v, const T v0, const T v1, const T len)
{
  T r = (v-v0)/(v1-v0)*len;
  return static_cast<unsigned>( r<0.5 ? 0xffffffff : r );
}

template<typename T>
__device__
T map( unsigned v, const T v0, const T v1, const T len)
{
  return static_cast<T>(v)/len*(v1-v0)+v0;
}

__device__ inline
unsigned char toColor(float v) {
  return static_cast<unsigned char>(255.0f*__saturatef(v));
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
T funcX(T t, T time, T talpha, T xk, T yk, const Parameters<T>& params) {
  switch(TFuncId) {
  case 0: return xk+talpha*cosf(params.t0+time+yk+cosf(params.t1+time+PI*xk));
  case 1: return t*yk+0.95f*xk-talpha*sinf(0.7f*yk);
  case 2: return xk - talpha*sinf( params.t1*yk + tanf(params.t0*yk) + time);
  case 3: return xk-talpha*sinf( params.t0+yk*time+tanf(3*yk+params.t1+time+sinf(2*yk+time)) );
  case 4: return xk+talpha*cosf( params.t0+yk*time+cosf(params.t1/(xk+1.0f)+time) );
  }
  return xk;
}

template<unsigned TFuncId, typename T>
inline __device__
T funcY(T t, T time, T talpha, T xk, T yk, const Parameters<T>& params) {
  switch(TFuncId) {
  case 0: return yk+talpha*cosf( params.t2+time+xk+cosf(params.t3+time+PI*yk));
  case 1: return t*xk+0.95f*yk+talpha*sinf(0.6f*xk);
  case 2: return yk - talpha*sinf( params.t3*xk + tanf(params.t2*xk) + time);
  case 3: return yk+talpha*sinf( params.t2+xk*time+tanf(3*xk+params.t3+time+sinf(2*xk+time)) );
  case 4: return yk+talpha*cosf( params.t2+xk*time+cosf(params.t3/(yk+1.0f)+time) );
  }
  return yk;
}

template<unsigned TFuncId, bool TPixelTrace, bool TSubSampling, typename T>
__global__
void d_generate_pattern(
  Data<T> _data,
  const Parameters<T> _params,
  const T _iteration_start,
  const T _iteration_end,
  const T _iteration_step_size,
  int _stage
  )
{
  unsigned i;
  unsigned px, py;
  T xk,yk, xk0, yk0;
  T width = _params.width;
  T height = _params.height;
  T t;
  T* density;
  density = _stage==-1 ? _data.buffer : _data.buffer+_stage*_params.n;

  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _params.n;
       i += blockDim.x * gridDim.x)
  {
    auto ix = (i%_params.width);
    auto iy = (i/_params.width);
    if(ix<0.5f*_params.border_width*_params.width || ix>(1.f-0.5f*_params.border_width)*_params.width
       || iy<0.5f*_params.border_width*_params.height || iy>(1.f-0.5f*_params.border_width)*_params.height)
      continue;

    xk0 = _data.buffer[i + 3*_params.n]; // already mapped into world space
    yk0 = _data.buffer[i + 4*_params.n];

    if(TSubSampling) {

      for(int dx=-1; dx<=1; ++dx) {
        for(int dy=-1; dy<=1; ++dy) {
          xk = xk0 + 0.5*T(dx)/_params.width*(_params.x1-_params.x0);
          yk = yk0 + 0.5*T(dy)/_params.height*(_params.y1-_params.y0);
          for(t=_iteration_start; t<_iteration_end; t+=_iteration_step_size) {
            xk = funcX<TFuncId>(t, _params.time, _params.talpha, xk, yk, _params);
            yk = funcY<TFuncId>(t, _params.time, _params.talpha, xk, yk, _params);
            px = unmap(xk, _params.x0, _params.x1, width);
            py = unmap(yk, _params.y0, _params.y1, height);

            if (px<_params.width && py<_params.height) {
              unsigned offset = px+py*_params.width;
              T v = _params.hit_value*powf(1.0f-t, _params.density_slope);
              if(dx!=0||dy!=0)
                v *= 0.5;
              if(_params.use_atomics)
                atomicAdd(density+offset, v); // just density
              else
                density[offset] += v;
            }
          } // for
        }
      }
    }else{
      xk = xk0;
      yk = yk0;

      if( TPixelTrace ) {
        if( ((i%_params.width)&_params.pixel_trace_divisor)==0
            && ((i/_params.width)&_params.pixel_trace_divisor)==0 ) {

          T time = sinf(_params.time) * _params.density_slope;
          for(t=_iteration_start; t<_iteration_end; t+=_iteration_step_size) {
            xk = funcX<TFuncId>(t, T(0.), time, xk, yk, _params);
            yk = funcY<TFuncId>(t, T(0.), time, xk, yk, _params);
            px = unmap(xk, _params.x0, _params.x1, width);
            py = unmap(yk, _params.y0, _params.y1, height);

            if (px<_params.width && py<_params.height) {
              unsigned offset = px+py*_params.width;
//                T v = 0.5*(T(t)/_params.iterations)+0.33;
              T v = 0.5*(T(t)/_iteration_end)+0.25;
              if(_params.use_atomics)
                atomicExch(density+offset, v);
              else
                density[offset] = v;
            }
          }
        }
      }
      else {
        for(t=_iteration_start; t<_iteration_end; t+=_iteration_step_size) {
          xk = funcX<TFuncId>(t, _params.time, _params.talpha, xk, yk, _params);
          yk = funcY<TFuncId>(t, _params.time, _params.talpha, xk, yk, _params);
          px = unmap(xk, _params.x0, _params.x1, width);
          py = unmap(yk, _params.y0, _params.y1, height);

          if (px<_params.width && py<_params.height) {
            unsigned offset = px+py*_params.width;
            T v = _params.hit_value*powf(1.0f-t, _params.density_slope);
            if(_params.use_atomics)
              atomicAdd(density+offset, v); // just density
            else
              density[offset] += v;
          }
        } // for
      } // TPixelTrace
    } // TSubSampling
    // only store current world position if in no-stage mode or in last stage
    if(_stage==-1 || _stage==2) {
      _data.buffer[i + 3*_params.n] = xk;
      _data.buffer[i + 4*_params.n] = yk;
    }
  }
}

template<Renderer TRenderer, bool TPixelTrace, typename T>
__global__
void d_render_to_image(
  uchar4 *_ptr,
  Data<T> _data,
  const Parameters<T> _params
  )
{
  unsigned j;
  for (j = blockIdx.x * blockDim.x + threadIdx.x;
       j < _params.n;
       j += blockDim.x * gridDim.x)
  {
    switch(TRenderer) {
    case Renderer::HSL:

      float h;
      float s;
      float l;

      if(TPixelTrace) {
        T v = 100*_params.hit_value*_data.buffer[j];
        if(_params.hue_end>=_params.hue_start)
          h = powf((_params.hue_end-_params.hue_start)*v, _params.hue_slope) + _params.hue_start;
        else
          h = powf((_params.hue_start-_params.hue_end)*v, _params.hue_slope) + _params.hue_end;
        if(h<0.0f)
          h += 1.0f;
        else if(h>1.0f)
          h -= 1.0f;
        s = __saturatef(_params.saturation_slope);
        l = __saturatef(v*_params.brightness_slope);
        if(_params.invert)
          l = 1.0f-l;
        hsl2rgb(h, s, l, _ptr[j]);

      } else {
        T v = _data.buffer[j];
        if(_params.hue_end>=_params.hue_start)
          h = powf((_params.hue_end-_params.hue_start)*__saturatef(v), _params.hue_slope) + _params.hue_start;
        else
          h = powf((_params.hue_start-_params.hue_end)*(__saturatef(v)), _params.hue_slope) + _params.hue_end;
        if(h<0.0f)
          h += 1.0f;
        else if(h>1.0f)
          h -= 1.0f;
        s = __saturatef(powf(v, _params.saturation_slope));
        l = __saturatef(powf(v, _params.brightness_slope));
        if(_params.invert)
          l = 1.0f-l;
        hsl2rgb(h, s, l, _ptr[j]);
      }

      break;
    case Renderer::QUILEZ:

      if(_params.invert) {
        _ptr[j].x = toColor( powf(powf(sqrtf(_data.buffer[j]), _params.hue_slope), _params.brightness_slope) );
        _ptr[j].y = toColor( powf(sqrtf(_data.buffer[j+_params.n]), _params.brightness_slope) );
        _ptr[j].z = toColor( powf(sqrtf(_data.buffer[j+2*_params.n]), _params.brightness_slope) );
      } else {
        _ptr[j].x = toColor( 1.-powf(powf(sqrtf(_data.buffer[j]), _params.hue_slope), _params.brightness_slope) );
        _ptr[j].y = toColor( 1.-powf(sqrtf(_data.buffer[j+_params.n]), _params.brightness_slope) );
        _ptr[j].z = toColor( 1.-powf(sqrtf(_data.buffer[j+2*_params.n]), _params.brightness_slope) );
      }
      break;
    default:
      T density = sqrt(_data.buffer[j]);//exp(-data.buffer[j])
      _ptr[j].x = toColor( 1.0-0.3*powf(density,0.4) );
      _ptr[j].y = toColor( 1.0-0.5*powf(density,1.0) );
      _ptr[j].z = toColor( 1.0-0.8*powf(density,1.4) );
      /*unsigned char d = 255*data.buffer[j];//exp(-data.buffer[j])
        ptr[j].x = d;
        ptr[j].y = d;
        ptr[j].z = d;*/
    }
  }
}

struct Launcher {

  template<bool TPixelTrace, bool TSubSampling, typename T>
  void launch(const Runner<T>& _runner,
              T it_start, T it_end, T it_step_size,
              int _stage
    ) {

    dim3 threads1d( 128 );
    dim3 blocks( 64*_runner.number_sm_ );

    switch(_runner.current_) {
    case Set::POPCORN0:
      d_generate_pattern<0, TPixelTrace, TSubSampling>
        <<<blocks, threads1d>>>(_runner.data_, _runner.params_, it_start, it_end, it_step_size, _stage);
      break;
    case Set::POPCORN1:
      d_generate_pattern<1, TPixelTrace, TSubSampling>
          <<<blocks, threads1d>>>(_runner.data_, _runner.params_, it_start, it_end, it_step_size, _stage);
      break;
    case Set::POPCORN2:
      d_generate_pattern<2, TPixelTrace, TSubSampling>
          <<<blocks, threads1d>>>(_runner.data_, _runner.params_, it_start, it_end, it_step_size, _stage);
      break;
    case Set::POPCORN3:
      d_generate_pattern<3, TPixelTrace, TSubSampling>
          <<<blocks, threads1d>>>(_runner.data_, _runner.params_, it_start, it_end, it_step_size, _stage);
      break;
    case Set::POPCORN4:
      d_generate_pattern<4, TPixelTrace, TSubSampling>
          <<<blocks, threads1d>>>(_runner.data_, _runner.params_, it_start, it_end, it_step_size, _stage);
      break;
    }
  }

  template<typename T>
  void operator()(const Runner<T>& _runner,
                  T it_start, T it_end, T it_step_size,
                  int _stage = -1
    ) {

    if(_runner.pixel_trace_) {
      if(_runner.sub_sampling_) {
        launch<true, true>( _runner, it_start, it_end, it_step_size, _stage);
      }else{
        launch<true, false>( _runner, it_start, it_end, it_step_size, _stage);
      }
    }else {
      if(_runner.sub_sampling_) {
        launch<false, true>( _runner, it_start, it_end, it_step_size, _stage);
      }else{
        launch<false, false>( _runner, it_start, it_end, it_step_size, _stage);
      }
    }
  }

  template<typename T>
  void operator()(const Runner<T>& _runner, uchar4* _pos) {

    dim3 threads1d( 128 );
    dim3 blocks( 64*_runner.number_sm_ );

    switch(_runner.renderer_) {
    case Renderer::DEFAULT:
      if(_runner.pixel_trace_)
        d_render_to_image<Renderer::DEFAULT, true>
          <<<blocks, threads1d>>>(_pos, _runner.data_, _runner.params_);
      else
        d_render_to_image<Renderer::DEFAULT, false>
          <<<blocks, threads1d>>>(_pos, _runner.data_, _runner.params_);
      break;
    case Renderer::HSL:
      if(_runner.pixel_trace_)
        d_render_to_image<Renderer::HSL, true>
          <<<blocks, threads1d>>>(_pos, _runner.data_, _runner.params_);
      else
        d_render_to_image<Renderer::HSL, false>
          <<<blocks, threads1d>>>(_pos, _runner.data_, _runner.params_);
      break;
    case Renderer::QUILEZ:
      if(_runner.pixel_trace_)
        d_render_to_image<Renderer::QUILEZ, true>
          <<<blocks, threads1d>>>(_pos, _runner.data_, _runner.params_);
      else
        d_render_to_image<Renderer::QUILEZ, false>
          <<<blocks, threads1d>>>(_pos, _runner.data_, _runner.params_);
      break;
    }
  }
};

template<typename T>
float Runner<T>::launch_kernel(cudaGraphicsResource* _dst, unsigned _iteration_offset) {

  size_t num_bytes;
  cudaError_t err;
  float ms = 0.0f;

  if(_iteration_offset==0)
    init_buffer();


  err=cudaGraphicsMapResources(1, &_dst, 0);
  if (err == cudaSuccess)  {
    uchar4* pos;
    const T it_start = _iteration_offset/T(params_.max_iterations);
    const T it_end = it_start + params_.iterations_per_run/T(params_.max_iterations);
    const T it_step_size = 1.0/T(params_.max_iterations);

    CHECK_CUDA(cudaEventRecord(custart));

    if(renderer_==Renderer::QUILEZ) {
      // red
      Launcher()(*this, it_start, it_end, it_step_size, 0);

      // green
      auto tmp = params_;
      params_.x0 = tmp.x0 * powf(0.999,tmp.saturation_slope);
      params_.x1 = tmp.x1 * powf(0.999,tmp.saturation_slope);
      params_.y0 = tmp.y0 * powf(0.999,tmp.saturation_slope);
      params_.y1 = tmp.y1 * powf(0.999,tmp.saturation_slope);
      params_.talpha = tmp.talpha * powf(1.7, tmp.hue_start);
      params_.density_slope = tmp.density_slope * powf(1.9, tmp.hue_start);
      //params_.hit_value = tmp.hit_value * powf(1.1, tmp.saturation_slope);
      Launcher()(*this, it_start, it_end, it_step_size, 1);

      // blue
      params_.x0 = tmp.x0 * powf(1.001,tmp.saturation_slope);
      params_.x1 = tmp.x1 * powf(1.001,tmp.saturation_slope);
      params_.y0 = tmp.y0 * powf(1.001,tmp.saturation_slope);
      params_.y1 = tmp.y1 * powf(1.001,tmp.saturation_slope);
      params_.talpha = tmp.talpha * powf(1.5, tmp.hue_end);
      params_.density_slope = tmp.density_slope * powf(1.7, tmp.hue_end);
      //params_.hit_value = tmp.hit_value * powf(0.8, tmp.saturation_slope);
      Launcher()(*this, it_start, it_end, it_step_size, 2);
      params_ = tmp;
    }else {
      Launcher()(*this, it_start, it_end, it_step_size, -1);
    }

    CHECK_CUDA(cudaEventRecord(cuend));

    CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&pos, &num_bytes, _dst));
    Launcher()(*this, pos);

    CHECK_CUDA( cudaEventSynchronize(cuend) );
    CHECK_CUDA( cudaEventElapsedTime(&ms, custart, cuend) );
  }
  CHECK_CUDA( cudaGraphicsUnmapResources(1, &_dst, 0));
  return ms;
}


template<typename T>
void Runner<T>::alloc_buffer()
{
  if(data_.buffer) {
    CHECK_CUDA( cudaFree(data_.buffer) );
    CHECK_CUDA( cudaEventDestroy(custart) );
    CHECK_CUDA( cudaEventDestroy(cuend) );
  }
  unsigned n = 5 * params_.n;
  CHECK_CUDA( cudaMalloc(&data_.buffer, n*sizeof(T)) );
  CHECK_CUDA( cudaEventCreate(&custart) );
  CHECK_CUDA( cudaEventCreate(&cuend) );
}

template<typename T>
void Runner<T>::init_buffer()
{
  int numSMs;
  int devId = 0;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);

  dim3 threads( 16, 16 );
  dim3 blocks( 32*numSMs );
  d_init_buffer<<<blocks, threads>>>(data_, params_);
  CHECK_CUDA( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
}

template<typename T>
void Runner<T>::cleanup_cuda()
{
  if(data_.buffer) {
    CHECK_CUDA( cudaFree(data_.buffer) );
    data_.buffer = 0;
  }
}

    template
    struct Runner<float>;
  }
}

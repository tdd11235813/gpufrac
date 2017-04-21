#include "cuda_globals.hpp"
#include "fractal_mccabe.cuh"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <curand_kernel.h>
/// @todo : variable seed, const restrict, grid-strides
// ---
#include <cudpp.h>


cudaEvent_t cstart, cend;
size_t d_satPitch = 0;
size_t d_satPitch_T = 0;
size_t d_satPitchInElements = 0;
size_t d_satPitchInElements_T = 0;
CUDPPHandle theCudpp;
CUDPPHandle scanPlan;
CUDPPConfiguration config = { CUDPP_SCAN,
                              CUDPP_ADD,
                              CUDPP_FLOAT, // @todo
                              CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE };
curandStatePhilox4_32_10_t *devStates;

// ---
// @todo T
static constexpr float dsinus[]   = {0,0.0, 0.0,0.866025,1,0.951057,0.866025};
static constexpr float dcosinus[] = {0,1.0,-1.0,-0.5,0,0.309017,0.5};



using namespace Fractal::McCabe;


__device__ inline
unsigned char toColor(float v) {
  return static_cast<unsigned char>(255.0f*__saturatef(v));
}

/// HSL [0:1] to RGB {0..255}, from http://stackoverflow.com/questions/4728581/hsl-image-adjustements-on-gpu
__device__
void hsl2rgb_mccabe( float hue, float sat, float lum, uchar4& color )
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
inline void find_min_max(T* begin, T* end, T *min, T *max){
    thrust::pair< thrust::device_ptr<T>, thrust::device_ptr<T> > tuple;
    tuple =
        thrust::minmax_element(
          thrust::device_ptr<T>(begin),
          thrust::device_ptr<T>(end)
        );

    *min = tuple.first[0];
    *max = tuple.second[0];
}


template<typename TRandState>
__global__ void d_setup_kernel(TRandState *state, uint n, int seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id>=n)
      return;
    curand_init(seed, id, 0, state+id);
}

/**
 * @todo to 4dim vec
 */
template<typename T, typename TRandState>
__global__ void d_initialize4( Data<T> _data,
                               const Parameters<T> _params,
                               TRandState *state,
                               uint n)
{
  using TVec = typename std::conditional<std::is_same<T,float>::value,float4,double4>::type;

  uint i=threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n)
    return;
  uint iy = i / _params.height;
  TRandState localState = state[i];
  TVec* pgrid4 = reinterpret_cast<TVec*>(_data.grid);
  TVec val4 = curand_uniform4(&localState);
  T gradient = static_cast<T>(iy)/_params.height;
  val4.x = (2.0*val4.x-1.0)*gradient;
  val4.y = (2.0*val4.y-1.0)*gradient;
  val4.z = (2.0*val4.z-1.0)*gradient;
  val4.w = (2.0*val4.w-1.0)*gradient;
  pgrid4[i] = val4;
  state[i] = localState;
}
/**
 *
 */
/*template<typename T, int TINVERT_MODE>
__global__ void d_reset_pattern( Data<T> data, curandState *state, int radius )
{
  int ix = (threadIdx.x + blockIdx.x * blockDim.x);
  int iy = (threadIdx.y + blockIdx.y * blockDim.y);
  int i = ix + iy * blockDim.x * gridDim.x;
//  uint i=threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= _params.n)
    return;
  if (ix<20 || iy<20 || ix+20>_params.width || iy+20>_params.height)
    return;
  curandState localState = state[i];
  T rnd = (T)curand_uniform(&localState);
  T gm = TINVERT_MODE==1?-1.:1.;
  if(rnd<T(0.0007)){
    for(int x=-radius; x<=radius; ++x)
      for(int y=-radius; y<=radius; ++y)
      {
        if(sqrtf(x*x+y*y)<float(radius))
          data.grid[ix+x + (iy+y)*_params.width] = gm;
      }
  }
  //state[i] = localState;
  }*/
/**
 *
 */
 /*template<typename T>
__global__ void d_clear_color(uchar4 *ptr, const Parameters<T> _params)
{
  uint i=threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= _params.n)
    return;
  ptr[i].x = 0;
  ptr[i].y = 0;
  ptr[i].z = 0;
}*/
/**
 *
 */
template<unsigned TSymmetry, typename T>
__device__ int getSymmetry(int i, int ix, int iy, const Parameters<T>& _params)
{
  if(TSymmetry == 2)
    return _params.n - 1 - i;
  else {
    float dx = ix - int(_params.width>>1);
    float dy = iy - int(_params.height>>1);
    int x2 = int(_params.width>>1)  + (int)(dx * dcosinus[TSymmetry] + dy * dsinus[TSymmetry]);
    int y2 = int(_params.height>>1) + (int)(dx * -dsinus[TSymmetry] + dy * dcosinus[TSymmetry]);
    int j = x2 + y2 * int(_params.width);

    return j<0 ? j+_params.n : j>=_params.n ? j-_params.n : j;
  }
}
/**
 *
 */
template<unsigned TSymmetry, typename T>
__global__ void d_symmetry(Data<T> _data, const Parameters<T> _params)
{
  uint ix = (threadIdx.x + blockIdx.x * blockDim.x);
  uint iy = (threadIdx.y + blockIdx.y * blockDim.y);
  uint i = ix + iy * _params.width;
  if (ix >= _params.width || iy >= _params.height)
    return;
  uint index = getSymmetry<TSymmetry>(i, ix, iy, _params);
  _data.grid[i] = _data.grid[i] * .94 + _data.backBuffer[index] * .06;
}
/**
 *
 */
template<typename T>
__global__ void d_blur_step2(T* target, T* backBuffer, T* sat,
                             size_t pitch, T* source, const Parameters<T> _params)
{
  uint ix = (threadIdx.x + blockIdx.x * blockDim.x);
  uint iy = (threadIdx.y + blockIdx.y * blockDim.y);
  uint i = ix + iy * _params.width;
  if (ix>=_params.width || iy>=_params.height)
    return;
  if(ix==0)
    target[i] = backBuffer[iy*_params.width] + source[iy*_params.width];
  else if(iy==0)
    target[i] = backBuffer[ix] + source[ix];
  else
  {
    target[i] = backBuffer[ix]
      + backBuffer[iy*_params.width]
      - backBuffer[0]
      + sat[ix + pitch*iy]
      -sat[(iy-1)*pitch]
      -sat[ix-1];
  }
}
/**
 *
 */
template<typename T>
__global__ void d_collect(T* to, T* buffer, uint radius, const Parameters<T> _params)
{
  uint ix = (threadIdx.x + blockIdx.x * blockDim.x);
  uint iy = (threadIdx.y + blockIdx.y * blockDim.y);
  uint i = ix + iy * _params.width;
//  uint i = ix + iy * _params.width;
  if (ix>=_params.width || iy>=_params.height)
    return;
  int minx = ix>radius ? ix-radius : 0;
  int maxx = min(ix + radius, _params.width - 1);
  int miny = iy>radius ? iy-radius : 0;
  int maxy = min(iy + radius, _params.height - 1);
  T area = 1.0/static_cast<T>((maxx - minx) * (maxy - miny));
  to[i] = ( buffer[maxy * _params.width + maxx]
            - buffer[maxy * _params.width + minx]
            - buffer[miny * _params.width + maxx]
            + buffer[miny * _params.width + minx]) * area;
}

#define M_GETBEST_J(elem, kk) if (TIsLevelZero || vabs.elem<var.elem) { \
        var4[j].elem = vabs.elem;\
        lvl4[j].elem = level;\
        dir4[j].elem = src.elem>tgt.elem;\
      }
/**
 *
 */
template<typename T, bool TIsLevelZero>
__global__ void d_getBest(Data<T> _data, T* target, T* source, int level, unsigned n)
{
  unsigned i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x)
  {
    T variation = fabs(source[i]-target[i]);
    if (TIsLevelZero || variation < _data.bestVariation[i]) {
      _data.bestVariation[i] = variation;
      _data.bestLevel[i] = level;
      _data.direction[i] = source[i] > target[i];
    }
  }
}

/**
 * @tparam TDir 0==default, 1==neg.dir, 2==pos.dir
 */
template<typename T, int TDir>
__global__ void d_advance(Data<T> _data, const Parameters<T> _params)
{
  int i;
  T delta = 100.0*_params.time_delta;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _params.n;
       i += blockDim.x * gridDim.x)
  {
      T curStep = delta*_data.stepSizes[_data.bestLevel[i]];
      if (TDir==1 || (!_data.direction[i] && TDir!=2) )
      {
        curStep = -curStep;
      }
      _data.grid[i] += curStep;
      _data.colorgrid[i] += curStep * _data.colorShift[_data.bestLevel[i]];
  }
}

template<typename T>
__global__ void d_dumpToImage(
      uchar4 *ptr,
      T* buffer,
      const Parameters<T> _params)
{
  uint i=threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= _params.n)
    return;
  ptr[i].x = 255*buffer[i];
}

template<bool TInvert, typename T>
__global__ void d_renderPattern(
      uchar4 *ptr,
      const Parameters<T> _params,
      T* grid,
      T* colorgrid,
      T gridmin,
      T gridrange,
      T colormin,
      T colorrange
    )
{
  uint i=threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= _params.n)
    return;
  float v,h,s,l;

  grid[i] = ((grid[i]-gridmin)/gridrange)-1.0;
  colorgrid[i] = ((colorgrid[i]-colormin)/colorrange)-1.0;

  if(TInvert)
  {
    v = 0.5f - 0.5f*grid[i];
    h = 0.5f - 0.5f*colorgrid[i];
  }else{
    v = 0.5f*grid[i]+0.5f;
    h = 0.5f*colorgrid[i]+0.5f;
  }
//(dmap(colorgrid[i], gmin, gmax, static_cast<float>(_params.hue_start), static_cast<float>(_params.hue_end)));
//  float v = dmap(grid[i], gmin, gmax, 0.0f, 1.0f);

//  int hue_offset = 255.0f * (_params.hue_start < 0.0 ? _params.hue_start+1.0 : _params.hue_start);

  v *= _params.density_slope;
  if(_params.hue_end>=_params.hue_start)
    h = powf((_params.hue_end-_params.hue_start)*__saturatef(h), _params.hue_slope) + _params.hue_start;
  else
    h = powf((_params.hue_start-_params.hue_end)*(__saturatef(h)), _params.hue_slope) + _params.hue_end;
  if(h<0.0f)
    h += 1.0f;
  else if(h>1.0f)
    h -= 1.0f;

  s = __saturatef(powf(v, _params.saturation_slope));
  l = __saturatef(powf(v, _params.brightness_slope));
  hsl2rgb_mccabe(h,s,l,ptr[i]);
}



template<typename T>
float Runner<T>::launch_kernel(cudaGraphicsResource* dst, bool advance) {
  uchar4* pos;
  size_t num_bytes;
  uint radius;
  cudaError_t err;
  float ms = 0.0f;
  int numSMs;

  T* backbuffer = data_.backBuffer;
  T* grid = data_.grid;
  T* blurbuffer = data_.blurBuffer;
  T* diffusion_right = data_.diffusionRight;
  T* diffusion_left  = data_.diffusionLeft;
  T* colorgrid  = data_.colorgrid;
  T* source = grid;
  T* target = diffusion_right;

  T gridmin, gridmax, gridrange;
  T colormin, colormax, colorrange;

  err=cudaGraphicsMapResources(1, &dst, 0);
  if (err == cudaSuccess)
  {
    CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(
        (void**)&pos, &num_bytes, dst));

    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    dim3 threads_1(128);
    dim3 blocks_1( (params_.n-1)/threads_1.x+1 );
    dim3 threads_2(32, 4);
    dim3 blocks_2((params_.width-1)/threads_2.x+1, (params_.height-1)/threads_2.y+1);
    dim3 blocks_1_sm( 32*numSMs );

    //d_clear_color<<<blocks_1,threads_1>>>(pos);

    /*
    int kradius;
    static long long unsigned ff = 0;
    ff += 1;
    if((ff&15)==15)
    {
      kradius = ff/250;
      if(kradius>5)
        kradius = 1;
      else
        kradius = 5-kradius;
      if(ff>7*250)
        ff = 0;
      if(M_CONTAINS_FLAG_TRUE(mode,M_INVERT_MODE))
        d_reset_pattern<T,1><<<blocks_2, threads_2>>>(*data_, devStates, kradius );
      else
        d_reset_pattern<T,0><<<blocks_2, threads_2>>>(*data_, devStates, kradius );
    }*/

    CHECK_CUDA(cudaEventRecord(cstart));

    if(advance) {
      if(params_.symmetry>0)
      {
        CHECK_CUDA(cudaMemcpy(backbuffer, grid, params_.n*sizeof(T), cudaMemcpyDeviceToDevice));
        switch(params_.symmetry) {
        case 1: d_symmetry<2><<<blocks_2, threads_2>>>(data_, params_); break;
        case 2: d_symmetry<3><<<blocks_2, threads_2>>>(data_, params_); break;
        case 3: d_symmetry<4><<<blocks_2, threads_2>>>(data_, params_); break;
        case 4: d_symmetry<5><<<blocks_2, threads_2>>>(data_, params_); break;
        }
      }

      for (int level = 0; level < data_.levels; level++)
      {
        radius = data_.radii_host[level];
        CHECK_CUDA(cudaMemcpy(backbuffer, blurbuffer, params_.n*sizeof(T), cudaMemcpyDeviceToDevice));
        if(level<=data_.blurlevels){
          blur_sat(blurbuffer, backbuffer, source);
        }

        d_collect<T><<<blocks_2,threads_2>>>(target, blurbuffer, radius, params_);
        if(level==0)
          d_getBest<T,true><<<blocks_1_sm, threads_1>>>(data_, target, source, level, params_.n);
        else
          d_getBest<T,false><<<blocks_1_sm, threads_1>>>(data_, target, source, level, params_.n);
        //        d_dumpToImage<<<blocks_1, threads_1>>>(pos, blurbuffer, params_);
        if((level&1)==0)
        {
          source = target;
          target = diffusion_left;
        }else{
          source = target;
          target = diffusion_right;
        }
      } // level


      if(params_.direction_mode==0)
        d_advance<T,0><<<blocks_1_sm, threads_1>>>(data_, params_);
      else if(params_.direction_mode==1)
        d_advance<T,1><<<blocks_1_sm, threads_1>>>(data_, params_);
      else
        d_advance<T,2><<<blocks_1_sm, threads_1>>>(data_, params_);

      find_min_max(grid, grid+params_.n, &gridmin, &gridmax);
      gridrange = 0.5*(gridmax - gridmin);
      find_min_max(colorgrid, colorgrid+params_.n, &colormin, &colormax);
      colorrange = 0.5*(colormax - colormin);
    }else{
      gridmin = -1.f;
      gridrange = 1.f;
      colormin = -1.f;
      colorrange = 1.f;
    }

    if(params_.invert)
      d_renderPattern<true>
        <<<blocks_1, threads_1>>>(pos, params_, grid, colorgrid, gridmin, gridrange, colormin, colorrange);
    else
      d_renderPattern<false>
        <<<blocks_1, threads_1>>>(pos, params_, grid, colorgrid, gridmin, gridrange, colormin, colorrange);

    CHECK_CUDA( cudaEventRecord(cend) );
    CHECK_CUDA( cudaEventSynchronize(cend) );
    CHECK_CUDA( cudaEventElapsedTime(&ms, cstart, cend) );
    CHECK_CUDA( cudaGraphicsUnmapResources(1, &dst, 0));
  }

  return ms;
}


template<typename T>
void Runner<T>::init_buffer(bool alloc) {
  if(params_.base<=1.0)
    throw std::runtime_error("McCabe: invalid base value (must be > 1.0)");

  if(alloc)
  {
    if(data_.backBuffer)
    {
      CHECK_CUDA( cudaFree(data_.backBuffer) );
      CHECK_CUDA( cudaFree(data_.grid) );
      CHECK_CUDA( cudaFree(data_.diffusionLeft) );
      CHECK_CUDA( cudaFree(data_.diffusionRight) );
      CHECK_CUDA( cudaFree(data_.blurBuffer) );
      CHECK_CUDA( cudaFree(data_.bestVariation) );
      CHECK_CUDA( cudaFree(data_.colorgrid) );
      CHECK_CUDA( cudaFree(data_.bestLevel) );
      CHECK_CUDA( cudaFree(data_.direction) );
      CHECK_CUDA(cudaFree(devStates));
      CHECK_CUDA(cudaEventDestroy(cstart));
      CHECK_CUDA(cudaEventDestroy(cend));
      cfin();
    }

    CHECK_CUDA(cudaEventCreate(&cstart));
    CHECK_CUDA(cudaEventCreate(&cend));

    CHECK_CUDA( cudaMalloc(&data_.backBuffer, params_.n*sizeof(T)) );
    CHECK_CUDA( cudaMalloc(&data_.grid, params_.n*sizeof(T)) );
    CHECK_CUDA( cudaMalloc(&data_.diffusionLeft, params_.n*sizeof(T)) );
    CHECK_CUDA( cudaMalloc(&data_.diffusionRight, params_.n*sizeof(T)) );
    CHECK_CUDA( cudaMalloc(&data_.blurBuffer, params_.n*sizeof(T)) );
    CHECK_CUDA( cudaMalloc(&data_.bestVariation, params_.n*sizeof(T)) );
    CHECK_CUDA( cudaMalloc(&data_.colorgrid, params_.n*sizeof(T)) );

    CHECK_CUDA( cudaMalloc(&data_.bestLevel, params_.n*sizeof(int)) );
    CHECK_CUDA( cudaMemset(data_.bestLevel, 0, params_.n*sizeof(int)));

    CHECK_CUDA( cudaMalloc(&data_.direction, params_.n*sizeof(bool)) );
    CHECK_CUDA( cudaMemset(data_.direction, 0, params_.n*sizeof(bool)));

    CHECK_CUDA( cudaMemset(data_.blurBuffer, 0.0, params_.n*sizeof(T)));
    CHECK_CUDA( cudaMemset(data_.diffusionLeft, 0.0, params_.n*sizeof(T)));
    CHECK_CUDA( cudaMemset(data_.diffusionRight, 0.0, params_.n*sizeof(T)));
    CHECK_CUDA( cudaMemset(data_.bestVariation, 0.0, params_.n*sizeof(T)));
    CHECK_CUDA( cudaMemset(data_.colorgrid, 0.0, params_.n*sizeof(T)));

    CHECK_CUDA(cudaMalloc(&devStates, params_.n * sizeof(curandStatePhilox4_32_10_t)));

  }

  int radius;
  // Pos Most Sig Bit - 1
  int new_levels = (int) (logf(max(params_.width,params_.height)) / logf(params_.base)) - 1;
  int new_blurlevels = (int) ((data_.levels+1.0f) * params_.blurFactor - 0.5f);
  if(new_blurlevels<0)
    new_blurlevels=0;

  if(data_.levels != new_levels)
  {
    data_.levels     = new_levels;
    data_.blurlevels = new_blurlevels;

    delete[] data_.radii_host;
    CHECK_CUDA( cudaFree(data_.radii) );
    CHECK_CUDA( cudaFree(data_.stepSizes) );
    CHECK_CUDA( cudaFree(data_.colorShift) );

    data_.radii_host = new unsigned[data_.levels];
    CHECK_CUDA( cudaMalloc(&data_.radii, data_.levels*sizeof(unsigned)) );
    CHECK_CUDA( cudaMalloc(&data_.stepSizes, data_.levels*sizeof(T)) );
    CHECK_CUDA( cudaMalloc(&data_.colorShift, data_.levels*sizeof(T)) );

  }

  auto* stepSizes   = new T[data_.levels];
  auto* colorShift  = new T[data_.levels];

  for(int i=0; i<data_.levels; ++i)
  {
    radius = (uint) pow(params_.base, i);
    data_.radii_host[i] = radius;
    stepSizes[i] = log(radius) * params_.stepScale + params_.stepOffset;
    colorShift[i] = ((i & 0x01) == 0 ? -1.0 : 1.0) * (data_.levels-i);
    //printf("i %d: r=%.3f, s=%.3f, c=%.3f\n",i, temps.radii[i], temps.stepSizes[i], temps.colorShift[i]);
    CHECK_CUDA(cudaMemcpy(data_.radii, data_.radii_host, data_.levels*sizeof(uint), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data_.stepSizes, stepSizes, data_.levels*sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data_.colorShift, colorShift, data_.levels*sizeof(T), cudaMemcpyHostToDevice));
  }
  delete[] stepSizes;
  delete[] colorShift;

  if(alloc) {
    uint n = (params_.n+3)/4;
    uint threads_1 = 128;
    uint blocks_1 = (n-1)/threads_1+1;
    d_setup_kernel<<<blocks_1, threads_1>>>(devStates, n, params_.seed);
    d_initialize4<T><<<blocks_1,threads_1>>>(data_, params_, devStates, n);
    CHECK_LAST("Initialization failed.");

    cinit();
  }
}
/**
 *
 */
template<typename T>
void Runner<T>::cleanup_cuda()
{
  if(data_.backBuffer)
  {
    CHECK_CUDA( cudaFree(data_.backBuffer) );
    CHECK_CUDA( cudaFree(data_.grid) );
    CHECK_CUDA( cudaFree(data_.diffusionLeft) );
    CHECK_CUDA( cudaFree(data_.diffusionRight) );
    CHECK_CUDA( cudaFree(data_.blurBuffer) );
    CHECK_CUDA( cudaFree(data_.bestVariation) );
    CHECK_CUDA( cudaFree(data_.colorgrid) );
    CHECK_CUDA( cudaFree(data_.bestLevel) );
    CHECK_CUDA( cudaFree(data_.direction) );
    CHECK_CUDA(cudaEventDestroy(cstart));
    CHECK_CUDA(cudaEventDestroy(cend));

    CHECK_CUDA(cudaFree(devStates));
    data_.backBuffer = nullptr;
    cfin();
  }
  if(data_.radii)
  {
    delete[] data_.radii_host;
    CHECK_CUDA( cudaFree(data_.radii) );
    CHECK_CUDA( cudaFree(data_.stepSizes) );
    CHECK_CUDA( cudaFree(data_.colorShift) );
    data_.radii = nullptr;
    data_.stepSizes = nullptr;
    data_.colorShift = nullptr;
    data_.radii_host = nullptr;
  }
  data_.levels = -1;
  params_.symmetry = 0;
}

// ---
// http://stackoverflow.com/questions/14174876/cuda-in-place-transpose-error
template<typename T, int TBlockSize>
__global__ void d_transpose(T* dst, T* src, int dstPitch, int srcPitch, int width, int height)
{
  __shared__ T block[TBlockSize][TBlockSize];

  int col = blockIdx.x * TBlockSize + threadIdx.x;
  int row = blockIdx.y * TBlockSize + threadIdx.y;

  if((col < width) && (row < height))
  {
    int tid_in = row * srcPitch + col;
    block[threadIdx.y][threadIdx.x] = src[tid_in];
  }

  __syncthreads();

  col = blockIdx.y * TBlockSize + threadIdx.x;
  row = blockIdx.x * TBlockSize + threadIdx.y;

  if((col < height) && (row < width))
  {
    int tid_out = row * dstPitch + col;
    dst[tid_out] = block[threadIdx.x][threadIdx.y];
  }
}

template<typename T>
void Runner<T>::cinit()
{
  unsigned width = params_.width;
  unsigned height = params_.height;
  size_t dpitch   = width  * sizeof(T);
  size_t dpitch_T = height * sizeof(T);

  CHECK_LAST("Before CUDA initialization");

  CHECK_CUDA( cudaMallocPitch( data_.SATs, &d_satPitch, dpitch, height));
  CHECK_CUDA( cudaMallocPitch( data_.SATs+1, &d_satPitch_T, dpitch_T, width));

  d_satPitchInElements   = d_satPitch   / sizeof(T);
  d_satPitchInElements_T = d_satPitch_T / sizeof(T);
  // Initialize CUDPP
  cudppCreate(&theCudpp);
}

template<typename T>
void Runner<T>::cfin()
{
  CHECK_CUDA(cudaFree(data_.SATs[0]));
  CHECK_CUDA(cudaFree(data_.SATs[1]));
  data_.SATs[0] = nullptr;
  data_.SATs[1] = nullptr;
  // shut down CUDPP
  if (CUDPP_SUCCESS != cudppDestroy(theCudpp))
  {
    printf("Error destroying CUDPP.\n");
  }
}

template<typename T>
void Runner<T>::blur_sat(T* _target,
                         T* _backBuffer,
                         T* _source)
{
  dim3 threads_2(16, 16);
  dim3 blocks_2  ((params_.width-1) / threads_2.x+1,
                  (params_.height-1) / threads_2.y+1);
  dim3 blocks_2_T((params_.height-1) / threads_2.x+1,
                  (params_.width-1) / threads_2.y+1);

  CHECK_CUDA(
      cudaMemcpy2D(data_.SATs[0], d_satPitch, _source, params_.width * sizeof(T),
                   params_.width * sizeof(T), params_.height,
                   cudaMemcpyDeviceToDevice));

  if (CUDPP_SUCCESS != cudppPlan(theCudpp, &scanPlan, config, params_.width, params_.height, d_satPitchInElements))
    fprintf(stderr, "Error creating CUDPPPlan.\n");

  // scan rows
  cudppMultiScan(scanPlan, data_.SATs[0], data_.SATs[0], params_.width, params_.height);

  // transpose so columns become rows
  d_transpose<T, 16> <<<blocks_2, threads_2, 0>>>(data_.SATs[1], data_.SATs[0],
                                                      d_satPitchInElements_T,
                                                      d_satPitchInElements,
                                                      params_.width,
                                                      params_.height);

  if (CUDPP_SUCCESS != cudppDestroyPlan(scanPlan))
    fprintf(stderr, "Error destroying CUDPPPlan.\n");
  if (CUDPP_SUCCESS != cudppPlan(theCudpp, &scanPlan, config, params_.height, params_.width, d_satPitchInElements_T))
    fprintf(stderr, "Error creating CUDPPPlan.\n");
  // scan columns
  cudppMultiScan(scanPlan, data_.SATs[1], data_.SATs[1], params_.height, params_.width);

  // transpose back
  d_transpose<T, 16> <<<blocks_2_T, threads_2, 0>>>(data_.SATs[0], data_.SATs[1],
                                                        d_satPitchInElements,
                                                        d_satPitchInElements_T,
                                                        params_.height,
                                                        params_.width);

  if (CUDPP_SUCCESS != cudppDestroyPlan(scanPlan))
    fprintf(stderr, "Error destroying CUDPPPlan.\n");
  d_blur_step2<<<blocks_2, threads_2>>>(_target, _backBuffer, data_.SATs[0],
                                        d_satPitchInElements, _source, params_);
}




// template
// void init_buffer<float>(Data<float>&, const Parameters<float>&, bool, int);
// template float launch_kernel(cudaGraphicsResource* dst,
//                              Data<float>& ddata,
//                              const Parameters<float>& params,
//                              bool advance,
//                              int direction_mode);
// template
// void cleanup_cuda(Data<float>& ddata);

namespace Fractal {
  namespace McCabe {
    template
    struct Runner<float>;
  }
}

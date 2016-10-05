#include "cuda_globals.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <curand_kernel.h>

// ---
#include <cudpp.h>


#define PTR_BACKBUFFER(buffer,n)      (buffer+0)
#define PTR_GRID(buffer,n)            (buffer+n)
#define PTR_DIFFUSION_LEFT(buffer,n)  (buffer+(2*n))
#define PTR_DIFFUSION_RIGHT(buffer,n) (buffer+(3*n))
#define PTR_BLUR(buffer,n)            (buffer+(4*n))
#define PTR_BESTVAR(buffer,n)         (buffer+(5*n))
#define PTR_COLORGRID(buffer,n)       (buffer+(6*n))

template<typename T>
void cinit(DataMc<T>&, uint width, uint height);
template<typename T>
void cfin(DataMc<T>&);
template<typename T>
void blur_sat(DataMc<T>& _data,
              T* target,
              T* backBuffer,
              T* source,
              const Parameters<T>& );

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
curandState *devStates;

// ---
// @todo T
static constexpr float dsinus[]   = {0,0.0, 0.0,0.866025,1,0.951057,0.866025};
static constexpr float dcosinus[] = {0,1.0,-1.0,-0.5,0,0.309017,0.5};

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


__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, id, 0, &state[id]);
}

template<typename T>
__device__ T dmap( T v, T a1, T b1, T a2, T b2 )
{
  return a2+(v-a1)/(b1-a1)*(b2-a2);
}
/**
 * @param h hue 0..255
 * @param h saturation 0..255
 * @param h brightness 0..255
 * @param out RGB Output (0..255)
 */
__device__ void hsb2rgb(unsigned char h, unsigned char s, unsigned char b, uchar4* out)
{
    float hh, ff, ss;
    unsigned char p,q,t;
    unsigned  i;

    if(s <= 0.0f) {       // < is bogus, just shuts up warnings
        out->x = b;
        out->y = b;
        out->z = b;
        return;
    }
    hh = 360.0f*0.00390625f*h;
    if(hh >= 360.0f)
      hh = 0.0f;
    hh *= 0.0166667f; //hh /= 60.0;
    i = (unsigned)hh;
    ff = hh - i; // fractional part
    ss = 0.00390625f*s; // 1/256

    p = b * (1.0f - ss);
    q = b * (1.0f - (ss * ff));
    t = b * (1.0f - (ss * (1.0f - ff)));

    switch(i)
    {
    case 0:
        out->x = b;
        out->y = t;
        out->z = p;
        break;
    case 1:
        out->x = q;
        out->y = b;
        out->z = p;
        break;
    case 2:
        out->x = p;
        out->y = b;
        out->z = t;
        break;

    case 3:
        out->x = p;
        out->y = q;
        out->z = b;
        break;
    case 4:
        out->x = t;
        out->y = p;
        out->z = b;
        break;
    case 5:
    default:
        out->x = b;
        out->y = p;
        out->z = q;
        break;
    }
}
/**
 *
 */
template<typename T>
__global__ void d_initialize( DataMc<T> _data,
                              const Parameters<T> _params,
                              curandState *state )
{
  uint i=threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= _params.n)
    return;
//  uint ix = i % _params.height;
  uint iy = i / _params.height;
  curandState localState = state[i];
  PTR_GRID(_data.buffer, _params.n)[i] = (T)iy/_params.height*curand_uniform(&localState);
  PTR_DIFFUSION_LEFT(_data.buffer, _params.n)[i] = 0.0;
  PTR_DIFFUSION_RIGHT(_data.buffer, _params.n)[i] = 0.0;
  PTR_BLUR(_data.buffer, _params.n)[i] = 0.0;
  state[i] = localState;
}
/**
 *
 */
/*template<typename T, int TINVERT_MODE>
__global__ void d_reset_pattern( DataMc<T> data, curandState *state, int radius )
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
  float dx = ix - int(_params.width>>1);
  float dy = iy - int(_params.height>>1);
  int x2 = int(_params.width>>1)  + (int)(dx * dcosinus[TSymmetry] + dy * dsinus[TSymmetry]);
  int y2 = int(_params.height>>1) + (int)(dx * -dsinus[TSymmetry] + dy * dcosinus[TSymmetry]);
  int j = x2 + y2 * int(_params.width);

  return j<0 ? j+_params.n : j>=_params.n ? j-_params.n : j;
}
/**
 *
 */
template<unsigned TSymmetry, typename T>
__global__ void symmetry(DataMc<T> _data, const Parameters<T> _params)
{
  uint ix = (threadIdx.x + blockIdx.x * blockDim.x);
  uint iy = (threadIdx.y + blockIdx.y * blockDim.y);
  uint i = ix + iy * blockDim.x * gridDim.x;
  if (i >= _params.n)
    return;
  uint index = getSymmetry<TSymmetry>(i, ix, iy, _params);
  T* grid_i = PTR_GRID(_data.buffer, _params.n)+i;
  *grid_i = *grid_i * .94 + PTR_BACKBUFFER(_data.buffer, _params.n)[index] * .06;
}
/**
 * boxfilter with summed area table (simple and slow)
 */
template<typename T>
__global__ void blur(DataMc<T> _data, T* source, const Parameters<T> _params)
{
  uint ix = (threadIdx.x + blockIdx.x * blockDim.x);
  uint iy = (threadIdx.y + blockIdx.y * blockDim.y);
  uint i = ix + iy * blockDim.x * gridDim.x;
//  uint i = ix + iy * _params.width;
  if (i >= _params.n)
    return;
  T sum = 0;
  T* blur = PTR_BLUR(_data.buffer, _params.n);
  T* backbuffer = PTR_BACKBUFFER(_data.buffer, _params.n);
  if(ix==0)
    blur[i] = backbuffer[iy*_params.width] + source[iy*_params.width];
  else if(iy==0)
    blur[i] = backbuffer[ix] + source[ix];
  else
  {
    sum = backbuffer[ix]
          + backbuffer[iy*_params.width]
          - backbuffer[0]
          - source[0]
          + source[ix]
          + source[iy*_params.width];
    for(uint jx=1;jx<=ix;++jx){
      for(uint jy=1;jy<=iy;++jy){
//    for(uint jx=1;jx<=ix;jx+=2)
//      for(uint jy=1;jy<=iy;jy+=2){
        sum += source[jx+jy*_params.width];
      }
    }
    blur[i] = sum;

  }
}
/**
 *
 */
template<typename T>
__global__ void blur_step2(T* target, T* backBuffer, T* sat,
                           size_t pitch, T* source, const Parameters<T> _params)
{
  uint ix = (threadIdx.x + blockIdx.x * blockDim.x);
  uint iy = (threadIdx.y + blockIdx.y * blockDim.y);
  uint i = ix + iy * blockDim.x * gridDim.x;
  if (i >= _params.n)
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
          + sat[i]
          -sat[(iy-1)*_params.width]
          -sat[ix-1];

  }
}
/**
 *
 */
template<typename T>
__global__ void collect(T* to, T* buffer, uint radius, const Parameters<T> _params)
{
  uint ix = (threadIdx.x + blockIdx.x * blockDim.x);
  uint iy = (threadIdx.y + blockIdx.y * blockDim.y);
  uint i = ix + iy * blockDim.x * gridDim.x;
//  uint i = ix + iy * _params.width;
  if (i >= _params.n)
    return;
  int minx = ix>radius ? ix-radius : 0;
  int maxx = min(ix + radius, _params.width - 1);
  int miny = iy>radius ? iy-radius : 0;
  int maxy = min(iy + radius, _params.height - 1);
  T area = 1.0/T((maxx - minx) * (maxy - miny));
  to[i] = ( buffer[maxy * _params.width + maxx]
            - buffer[maxy * _params.width + minx]
            - buffer[miny * _params.width + maxx]
            + buffer[miny * _params.width + minx]) * area;
}
/**
 * @todo use other functions
 */
template<typename T>
__global__ void getBest(DataMc<T> _data, T* target, T* source, int level, unsigned n)
{
  uint i=threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n)
    return;

  T variation = source[i] - target[i];
  if(variation<0.0)
    variation=-variation;

  T* bestVariation = PTR_BESTVAR(_data.buffer, n);
  if (level == 0 || variation < bestVariation[i]) {
    bestVariation[i] = variation;
    _data.bestLevel[i] = level;
    _data.direction[i] = source[i] > target[i];
  }
}
/**
 *
 */
template<typename T>
__global__ void advance(DataMc<T> _data, unsigned n)
{
  uint i=threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n)
    return;
  T curStep = _data.stepSizes[_data.bestLevel[i]];
  T* grid = PTR_GRID(_data.buffer, n);
  T* colorgrid = PTR_COLORGRID(_data.buffer, n);
  if (_data.direction[i])
  {
    grid[i] += curStep;
    colorgrid[i] += curStep * _data.colorShift[_data.bestLevel[i]];
  }
  else {
    grid[i] -= curStep;
    colorgrid[i] -= curStep * _data.colorShift[_data.bestLevel[i]];
  }
}

template<bool TInvert, typename T>
__global__ void renderPattern(
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
  T gmin, gmax;
  if(TInvert)
  {
    gmin = -1.;
    gmax = +1.;
  }else{
    gmin = +1.;
    gmax = -1.;
  }

  grid[i] = ((grid[i]-gridmin)/gridrange)-1.0;
  colorgrid[i] = ((colorgrid[i]-colormin)/colorrange)-1.0;
  T hue_offset = _params.hue_start < 0.0 ? _params.hue_start+1.0 : _params.hue_start;
  // @todo hue coloring
  unsigned char h = int(dmap(colorgrid[i], gmin, gmax, T(0.0), T(127.0)) + 255*hue_offset) & 0xff;
  unsigned char b = dmap(grid[i], gmin, gmax, T(0.0), T(255.0));
  unsigned char s = (255-b)>>1;//(0.5f*(255.0f-b));
  hsb2rgb(h,s,b,&ptr[i]);
}
/**
 *
 */
template<typename T>
float launch_kernel(cudaGraphicsResource* dst,
                   DataMc<T>& _data,
                   const Parameters<T>& _params)
{
  dim3 threads_1(128);
  dim3 blocks_1( (_params.n-1)/threads_1.x+1 );
  dim3 threads_2(16,16);
  dim3 blocks_2(_params.width/threads_2.x, _params.height/threads_2.y);
  uchar4* pos;
  size_t num_bytes;
  uint radius;
  cudaError_t err;
  T* backbuffer = PTR_BACKBUFFER(_data.buffer, _params.n);
  T* grid = PTR_GRID(_data.buffer, _params.n);
  T* blurbuffer = PTR_BLUR(_data.buffer, _params.n);
  T* diffusion_right = PTR_DIFFUSION_RIGHT(_data.buffer, _params.n);
  T* diffusion_left  = PTR_DIFFUSION_LEFT(_data.buffer, _params.n);
  T* colorgrid  = PTR_COLORGRID(_data.buffer, _params.n);
  T* source = grid;
  T* target = diffusion_right;

  T gridmin, gridmax, gridrange;
  T colormin, colormax, colorrange;

  err=cudaGraphicsMapResources(1, &dst, 0);
  if (err == cudaSuccess)
  {
    CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(
        (void**)&pos, &num_bytes, dst));
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
        d_reset_pattern<T,1><<<blocks_2, threads_2>>>(*_data, devStates, kradius );
      else
        d_reset_pattern<T,0><<<blocks_2, threads_2>>>(*_data, devStates, kradius );
    }*/

    if(_data.symmetry>1)
    {
      CHECK_CUDA(cudaMemcpy(backbuffer, grid, _params.n*sizeof(T), cudaMemcpyDeviceToDevice));
      symmetry<0><<<blocks_2, threads_2>>>(_data, _params);
    }

    for (int level = 0; level < _data.levels; level++)
    {
      radius = _data.radii_host[level];
      CHECK_CUDA(cudaMemcpy(backbuffer, blurbuffer, _params.n*sizeof(T), cudaMemcpyDeviceToDevice));
      if(level<=_data.blurlevels){
//        blur<T><<<blocks_2,threads_2>>>(*_data, source);
        blur_sat(_data, blurbuffer, backbuffer, source, _params);
      }

      collect<T><<<blocks_2,threads_2>>>(target, blurbuffer, radius, _params);
      getBest<T><<<blocks_1, threads_1>>>(_data, target, source, level, _params.n);

      if((level&1)==0)
      {
        source = target;
        target = diffusion_left;
      }else{
        source = target;
        target = diffusion_right;
      }
    } // level


    advance<T><<<blocks_1, threads_1>>>(_data, _params.n);

    find_min_max(grid, grid+_params.n, &gridmin, &gridmax);
    gridrange = 0.5*(gridmax - gridmin);
    find_min_max(colorgrid, colorgrid+_params.n, &colormin, &colormax);
    colorrange = 0.5*(colormax - colormin);
    if(_params.invert)
      renderPattern<true>
        <<<blocks_1, threads_1>>>(pos, _params, grid, colorgrid, gridmin, gridrange, colormin, colorrange);
    else
      renderPattern<false>
        <<<blocks_1, threads_1>>>(pos, _params, grid, colorgrid, gridmin, gridrange, colormin, colorrange);

    CHECK_CUDA( cudaGraphicsUnmapResources(1, &dst, 0));
  }
  return 0.0f;
}
/**
 *
 */
template<typename T>
void upload_parameters(
  DataMc<T>& _data,
  const Parameters<T>& _params)
{
  int radius;
  // Pos Most Sig Bit - 1
  int new_levels = (int) (
    logf(max(_params.width,_params.height)) / logf(_data.base)) - 1;
  int new_blurlevels = (int) ((_data.levels+1.0f) * _data.blurFactor - 0.5f);
  if(new_blurlevels<0)
    new_blurlevels=0;

  if(_data.levels != new_levels)
  {
    _data.levels     = new_levels;
    _data.blurlevels = new_blurlevels;

    delete[] _data.radii_host;
    CHECK_CUDA( cudaFree(_data.radii) );
    CHECK_CUDA( cudaFree(_data.stepSizes) );
    CHECK_CUDA( cudaFree(_data.colorShift) );

    _data.radii_host = new unsigned[_data.levels];
    CHECK_CUDA( cudaMalloc((void**)(&_data.radii), _data.levels*sizeof(unsigned)) );
    CHECK_CUDA( cudaMalloc((void**)(&_data.stepSizes), _data.levels*sizeof(T)) );
    CHECK_CUDA( cudaMalloc((void**)(&_data.colorShift), _data.levels*sizeof(T)) );

  }

  auto* stepSizes   = new T[_data.levels];
  auto* colorShift  = new T[_data.levels];

  for(int i=0; i<_data.levels; ++i)
  {
    radius = (uint) pow(_data.base, i);
    _data.radii_host[i] = radius;
    stepSizes[i] = log(radius) * _data.stepScale + _data.stepOffset;
    colorShift[i] = ((i & 0x01) == 0 ? -1.0 : 1.0) * (_data.levels-i);
    //printf("i %d: r=%.3f, s=%.3f, c=%.3f\n",i, temps.radii[i], temps.stepSizes[i], temps.colorShift[i]);
    CHECK_CUDA(cudaMemcpy(_data.radii, _data.radii_host, _data.levels*sizeof(uint), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(_data.stepSizes, stepSizes, _data.levels*sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(_data.colorShift, colorShift, _data.levels*sizeof(T), cudaMemcpyHostToDevice));
  }

  delete[] stepSizes;
  delete[] colorShift;
}

/**
 *
 */
template<typename T>
void init_buffer(DataMc<T>& _data,
                 const Parameters<T>& _params,
                 bool alloc)
{
  uint threads_1 = 128;
  uint blocks_1 = (_params.n-1)/threads_1+1;

  if(alloc)
  {
    if(_data.buffer)
    {
      CHECK_CUDA( cudaFree(_data.buffer) );
      CHECK_CUDA( cudaFree(_data.bestLevel) );
      CHECK_CUDA( cudaFree(_data.direction) );
      CHECK_CUDA(cudaFree(devStates));
      cfin(_data);
    }
    CHECK_CUDA( cudaMalloc((void**)(&_data.buffer), 7*_params.n*sizeof(T)) );

    CHECK_CUDA( cudaMalloc((void**)(&_data.bestLevel), _params.n*sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**)(&_data.direction), _params.n*sizeof(bool)) );

    CHECK_CUDA( cudaMemset(PTR_BLUR(_data.buffer, _params.n), 0.0, _params.n*sizeof(T)));
    CHECK_CUDA( cudaMemset(PTR_BESTVAR(_data.buffer, _params.n), 0.0, _params.n*sizeof(T)));
    CHECK_CUDA( cudaMemset(PTR_COLORGRID(_data.buffer, _params.n), 0.0, _params.n*sizeof(T)));

    CHECK_CUDA(cudaMalloc((void **)&devStates, _params.n * sizeof(curandState)));

/*
    if(dtemps.radii)
    {
      delete[] temps.radii;
      delete[] temps.stepSizes;
      delete[] temps.colorShift;
      CHECK_CUDA( cudaFree(dtemps.radii) );
      CHECK_CUDA( cudaFree(dtemps.stepSizes) );
      CHECK_CUDA( cudaFree(dtemps.colorShift) );
    }

    temps.radii       = new uint[temps.levels];
    temps.stepSizes   = new float[temps.levels];
    temps.colorShift  = new float[temps.levels];
    CHECK_CUDA( cudaMalloc((void**)(&dtemps.radii), temps.levels*sizeof(uint)) );
    CHECK_CUDA( cudaMalloc((void**)(&dtemps.stepSizes), temps.levels*sizeof(T)) );
    CHECK_CUDA( cudaMalloc((void**)(&dtemps.colorShift), temps.levels*sizeof(T)) );
    */
  }

  //float cosinus[] = { 0, cos(TWO_PI/1), cos(TWO_PI/2),  cos(TWO_PI/3),  cos(TWO_PI/4),  cos(TWO_PI/5),  cos(TWO_PI/6) };
  //float sinus[] =  { 0, sin(TWO_PI/1), sin(TWO_PI/2),  sin(TWO_PI/3),  sin(TWO_PI/4),  sin(TWO_PI/5),  sin(TWO_PI/6) };
  //CHECK_CUDA(cudaMemcpyToSymbol(dcosinus,cosinus,7*sizeof(float),0,cudaMemcpyHostToDevice));
  //CHECK_CUDA(cudaMemcpyToSymbol(dsinus,sinus,7*sizeof(float),0,cudaMemcpyHostToDevice));

  setup_kernel<<<blocks_1, threads_1>>>(devStates);
  d_initialize<T><<<blocks_1,threads_1>>>(_data, _params, devStates);
  CHECK_LAST("Initialization failed.");

  cinit(_data, _params.width, _params.height);
}
/**
 *
 */
template<typename T>
void cleanup_cuda(DataMc<T>& _data)
{
  if(_data.buffer)
  {
    CHECK_CUDA( cudaFree(_data.buffer) );
    CHECK_CUDA( cudaFree(_data.bestLevel) );
    CHECK_CUDA( cudaFree(_data.direction) );

    CHECK_CUDA(cudaFree(devStates));
    _data.buffer = nullptr;
    cfin(_data);
  }
  if(_data.radii)
  {
    delete[] _data.radii_host;
    CHECK_CUDA( cudaFree(_data.radii) );
    CHECK_CUDA( cudaFree(_data.stepSizes) );
    CHECK_CUDA( cudaFree(_data.colorShift) );
    _data.radii = nullptr;
    _data.stepSizes = nullptr;
    _data.colorShift = nullptr;
    _data.radii_host = nullptr;
  }
  _data.levels = -1;
  _data.symmetry = 0;
}

// ---

template <typename T, int block_width, int block_height>
__global__ void transpose(T *out,
                          T *in,
                          size_t pitch_out,
                          size_t pitch_in,
                          size_t width,
                          size_t height)
{
    __shared__ T block[block_width*block_height];

    unsigned int xBlock = blockDim.x * blockIdx.x;
    unsigned int yBlock = blockDim.y * blockIdx.y;
    unsigned int xIndex = xBlock + threadIdx.x;
    unsigned int yIndex = yBlock + threadIdx.y;
    unsigned int index_out, index_transpose;

    if (xIndex < width && yIndex < height)
    {
        // load block into smem
        unsigned int index_in  =
                pitch_in*(yBlock + threadIdx.y) +
                xBlock + threadIdx.x;

        unsigned int index_block = threadIdx.y * block_width + threadIdx.x;
        block[index_block] = in[index_in];

        index_transpose = threadIdx.x*block_width + threadIdx.y;

        index_out = pitch_out * (xBlock + threadIdx.y) +
            yBlock + threadIdx.x;
    }

    __syncthreads();

    if (xIndex < width && yIndex < height)
    {
        // write it out (transposed) into the new location
        out[index_out] = block[index_transpose];
    }
}

template<typename T>
void cinit(DataMc<T>& _data, uint width, uint height)
{
    size_t dpitch   = width  * sizeof(T);
    size_t dpitch_T = height * sizeof(T);

    CHECK_LAST("Before CUDA initialization");

    CHECK_CUDA( cudaMallocPitch( (void**) _data.SATs, &d_satPitch, dpitch, height));
    CHECK_CUDA( cudaMallocPitch( (void**) _data.SATs+1, &d_satPitch_T, dpitch_T, width));

    /*CHECK_CUDA( cudaMemset2D(SATs[0], d_satPitch, 0.0, width*sizeof(T), height));
    CHECK_CUDA( cudaMemset2D(SATs[1], d_satPitch_T, 0.0, height*sizeof(T), width));*/

    d_satPitchInElements   = d_satPitch   / sizeof(T);
    d_satPitchInElements_T = d_satPitch_T / sizeof(T);
    // Initialize CUDPP
    cudppCreate(&theCudpp);
}

template<typename T>
void cfin(DataMc<T>& _data)
{
  CHECK_CUDA(cudaFree(_data.SATs[0]));
  CHECK_CUDA(cudaFree(_data.SATs[1]));
  _data.SATs[0] = nullptr;
  _data.SATs[1] = nullptr;
  // shut down CUDPP
  if (CUDPP_SUCCESS != cudppDestroy(theCudpp))
  {
      printf("Error destroying CUDPP.\n");
  }
}

template<typename T>
void blur_sat(DataMc<T>& _data,
              T* _target,
              T* _backBuffer,
              T* _source,
              const Parameters<T>& _params)
{
  dim3 threads_2(16, 16);
  dim3 blocks_2  (_params.width / threads_2.x,
                  _params.height / threads_2.y);
  dim3 blocks_2_T(_params.height / threads_2.x,
                  _params.width / threads_2.y);

  CHECK_CUDA(
      cudaMemcpy2D(_data.SATs[0], d_satPitch, _source, _params.width * sizeof(T),
                   _params.width * sizeof(T), _params.height,
                   cudaMemcpyDeviceToDevice));

  if (CUDPP_SUCCESS != cudppPlan(theCudpp, &scanPlan, config, _params.width, _params.height, d_satPitchInElements))
    fprintf(stderr, "Error creating CUDPPPlan.\n");

  // scan rows
  cudppMultiScan(scanPlan, _data.SATs[0], _data.SATs[0], _params.width, _params.height);

  // transpose so columns become rows
  transpose<T, 16, 16> <<<blocks_2, threads_2, 0>>>(_data.SATs[1], _data.SATs[0],
                                                          d_satPitchInElements_T,
                                                          d_satPitchInElements,
                                                          _params.width,
                                                          _params.height);

  if (CUDPP_SUCCESS != cudppDestroyPlan(scanPlan))
    fprintf(stderr, "Error destroying CUDPPPlan.\n");
  if (CUDPP_SUCCESS != cudppPlan(theCudpp, &scanPlan, config, _params.height, _params.width, d_satPitchInElements_T))
    fprintf(stderr, "Error creating CUDPPPlan.\n");
  // scan columns
  cudppMultiScan(scanPlan, _data.SATs[1], _data.SATs[1], _params.height, _params.width);

  // transpose back
  transpose<T, 16, 16> <<<blocks_2_T, threads_2, 0>>>(_data.SATs[0], _data.SATs[1],
                                                            d_satPitchInElements,
                                                            d_satPitchInElements_T,
                                                            _params.height,
                                                            _params.width);

  if (CUDPP_SUCCESS != cudppDestroyPlan(scanPlan))
    fprintf(stderr, "Error destroying CUDPPPlan.\n");
  blur_step2<<<blocks_2, threads_2>>>(_target, _backBuffer, _data.SATs[0],
                                      d_satPitchInElements, _source, _params);
}



template
void upload_parameters<float>(DataMc<float>&, const Parameters<float>&);
template
void init_buffer<float>(DataMc<float>&, const Parameters<float>&, bool);
template float launch_kernel(cudaGraphicsResource* dst,
                             DataMc<float>& ddata,
                             const Parameters<float>& params);
template
void cleanup_cuda(DataMc<float>& ddata);
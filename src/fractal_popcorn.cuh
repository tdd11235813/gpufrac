#ifndef FRACTAL_POPCORN_H
#define FRACTAL_POPCORN_H

#include <cuda_gl_interop.h>

namespace Fractal {
  namespace Popcorn {

    template<typename T>
    struct Data
    {
      T* buffer = nullptr;
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
      T hit_value = 0.01;
      T time = 0.0; // elapsed
      T time_delta = 0.0; // time delta to last frame
      unsigned width=800;
      unsigned height=800;
      unsigned n=0;
      unsigned max_iterations=64;
      unsigned iterations=64;
      unsigned iterations_per_run=32;
      unsigned pixel_trace_divisor=3;
      T hue_start = 0.0;
      T hue_end   = 0.3;
      T hue_slope = 1.45;
      T density_slope = 1.0;
      T brightness_slope = 1.0;
      T saturation_slope = 1.0;
      float border_width = 0.0f;
      bool use_atomics = true;
      bool invert = false;
    };

    enum class Set {
      POPCORN0=0,
      POPCORN1,
      POPCORN2,
      POPCORN3,
      POPCORN4
    };

    enum class Renderer {
      DEFAULT=0,
      HSL,
      QUILEZ
    };

    template<typename T>
    struct Runner {
      Data<T> data_;
      Parameters<T> params_;
      Set current_ = Set::POPCORN0;
      Renderer renderer_ = Renderer::DEFAULT;
      bool pixel_trace_ = false;
      bool sub_sampling_ = false;
      int number_sm_ = 0;
      int dev_id_ = 0;

      Runner() {
        cudaDeviceGetAttribute(&number_sm_, cudaDevAttrMultiProcessorCount, dev_id_);
      }

      float launch_kernel(cudaGraphicsResource* dst, unsigned);

      void alloc_buffer();

      void init_buffer();

      void cleanup_cuda();

      void reset() {
        params_ = Parameters<T>();
      }
    };

  }
}

#endif /* FRACTAL_POPCORN_H */

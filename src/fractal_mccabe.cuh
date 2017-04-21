#ifndef FRACTAL_MCCABE_H
#define FRACTAL_MCCABE_H

#include <cuda_gl_interop.h>

namespace Fractal {
  namespace McCabe {

    template<typename T>
    struct Data
    {
      T* backBuffer = nullptr;
      T* grid = nullptr;
      T* diffusionLeft = nullptr;
      T* diffusionRight = nullptr;
      T* blurBuffer = nullptr;
      T* bestVariation = nullptr;
      T* colorgrid = nullptr;
      T* stepSizes = nullptr;
      T* colorShift = nullptr;
      T* SATs[2] = {nullptr, nullptr};
      int* bestLevel = nullptr;
      bool* direction = nullptr;
      unsigned *radii = nullptr;
      unsigned *radii_host = nullptr;

      int levels = -1;
      int blurlevels = 1;
    };

    /// Parameters
    template<typename T>
    struct Parameters
    {
      unsigned width=800;
      unsigned height=800;
      unsigned n=0;

      float blurFactor = 1.0f;
      //T scale; // space zoom (scale)
      T base = 2.0;
      T stepScale = 0.002;
      T stepOffset = 0.002;

      T time = 0.0; // elapsed
      T time_delta = 0.0; // time delta to last frame

      T hue_start = 0.0;
      T hue_end   = 0.3;
      T hue_slope = 1.45;
      T density_slope = 1.0;
      T brightness_slope = 1.0;
      T saturation_slope = 1.0;
      int direction_mode = 0;
      int symmetry = 0;
      int seed = 1234;
      bool invert = false;
    };


    template<typename T>
    struct Runner {
      Data<T> data_;
      Parameters<T> params_;

      float launch_kernel(cudaGraphicsResource* dst, bool advance);

      void init_buffer(bool alloc);

      void cleanup_cuda();

      void reset() {
        params_ = Parameters<T>();
      }

      void cinit();

      void cfin();

      void blur_sat(T* target,
                    T* backBuffer,
                    T* source);

    };

  }
}

#endif /* FRACTAL_MCCABE_H */

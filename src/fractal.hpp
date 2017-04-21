#ifndef FRACTAL_H
#define FRACTAL_H

#include "fractal_popcorn.cuh"
#include "fractal_mccabe.cuh"


namespace Fractal {

  enum class Set {
    POPCORN=0,
    MCCABE,
    _COUNT
  };

  template<typename T>
  struct Fractals
  {
    Set current_ = Set::POPCORN;

    Popcorn::Runner<T> popcorn_;
    McCabe::Runner<T> mccabe_;



    void reset_time() {
      switch(current_) {
      case Set::MCCABE:
        mccabe_.params_.time = 0;
        break;
      case Set::POPCORN:
        popcorn_.params_.time = 0;
      }

    }

    void reset_current() {
      unsigned w, h;
      switch(current_) {
      case Set::MCCABE:
        w = mccabe_.params_.width;
        h = mccabe_.params_.height;
        mccabe_.reset();
        break;
      case Set::POPCORN:
        w = popcorn_.params_.width;
        h = popcorn_.params_.height;
        popcorn_.reset();
      }
      // set to previous image size
      set_size(w,h);
    }

    unsigned width() const {
      switch(current_) {
      case Set::MCCABE: return mccabe_.params_.width;
      case Set::POPCORN: return popcorn_.params_.width;
      }
    }

    unsigned height() const {
      switch(current_) {
      case Set::MCCABE: return mccabe_.params_.height;
      case Set::POPCORN: return popcorn_.params_.height;
      }
    }


    void set_size(unsigned w, unsigned h) {
      switch(current_) {
      case Set::MCCABE:
        mccabe_.params_.n = w*h;
        mccabe_.params_.width = w;
        mccabe_.params_.height = h;
        break;
      case Set::POPCORN:
        popcorn_.params_.n = w*h;
        popcorn_.params_.width = w;
        popcorn_.params_.height = h;
      }
    }

    void cleanup() {
      switch(current_) {
      case Set::MCCABE:
        mccabe_.cleanup_cuda();
        break;
      case Set::POPCORN:
        popcorn_.cleanup_cuda();
        break;
      }
    }
  };

}

#endif /* FRACTAL_H */

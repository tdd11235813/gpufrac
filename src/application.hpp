/*
 * application.h
 *
 *  Created on: 17.09.2016
 *      Author: mwerner
 */

#ifndef APPLICATION_H_
#define APPLICATION_H_

#include "gl_globals.hpp"
#include "cuda_globals.hpp" // ... also simulation settings
#include "shader.hpp"
#include "image.hpp"

#include <mutex>
#include <array>

#ifdef Success
#undef Success
#endif
#ifdef None
#undef None
#endif
#include <nanogui/nanogui.h>

enum class Resolution {
  SMALL=0,
  MEDIUM,
  SD,
  HD,
  HD2,
  MCCABE_SMALL,
  MCCABE_MEDIUM,
  MCCABE_HIGH
};

template<typename T>
class Application : public nanogui::Screen
{
public:
  Application();
  virtual
  ~Application();

  virtual void draw(NVGcontext *ctx) override;
  virtual bool keyboardEvent(int key, int scancode, int action, int modifiers) override;
  virtual void drawContents() override;

private:
  void create_quad();
  void initCuda();
  void cleanup();
  void create_shader();
  void create_buffers();
  void runCuda();
  void computeFPS();
  void recompute();

  template<typename U, typename V>
  void update_value(U& target, V source) {
    if(target != source) {
      target = static_cast<U>(source);
      recompute();
    }
  }

  virtual bool mouseMotionEvent(const Eigen::Vector2i &p,
                                const Eigen::Vector2i &rel,
                                int button, int modifiers) override;
  virtual bool mouseButtonEvent(const Eigen::Vector2i &p,
                                int button, bool down, int modifiers) override;
  virtual bool scrollEvent(const Eigen::Vector2i &p, const Eigen::Vector2f &rel) override;
  virtual bool resizeEvent(const Eigen::Vector2i &) override;

private:
  GLuint quadFinalVBOId_ = 0, quadIBOId_ = 0, vertexArrayQuad_ = 0;
  GLuint texId_ = 0;
  GLuint imagePBO_ = 0;
  cudaGraphicsResource_t resource_ = 0;
  ShaderManager shader_;

  bool app_started_ = false;
  bool rescale_ = false;
  bool reset_buffer_ = false;
  bool screenshot_ = false;
  bool recompute_ = false;
  bool mousePressed_ = false;
  unsigned tmp_texWidth_;
  unsigned tmp_texHeight_;
  //    std::mutex mutex_max_iterations_;

  Parameters<T> parameters_;
  Data<T> ddata_;
  DataMc<T> ddata_mc_;
  UserSetting settings_;
  double fps_ = 0.0;
  float ms_kernel_ = 0.0f;
  Resolution resolution_;
  unsigned iterations_offset_;

  nanogui::ref<nanogui::Window> window_;
  nanogui::ref<nanogui::Window> window_status_;
  nanogui::ref<nanogui::Window> window_shading_;
  nanogui::Button* btnAnimate_ = nullptr;
  nanogui::Label* labelStatus_ = nullptr;
  nanogui::Label* labelPosition_ = nullptr;
  std::array<nanogui::FloatBox<double>*, 4> gui_coef_ = {{nullptr}};
  std::array<nanogui::FloatBox<double>*, 4> gui_zoom_ = {{nullptr}};
  nanogui::FloatBox<double>* gui_talpha_ = nullptr;
  nanogui::FloatBox<double>* gui_add_ = nullptr;
  nanogui::IntBox<unsigned>* gui_texWidth_ = nullptr;
  nanogui::IntBox<unsigned>* gui_texHeight_ = nullptr;
  nanogui::Slider* gui_iterations_ = nullptr;
};



#endif /* APPLICATION_H_ */

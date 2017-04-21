#ifndef APPLICATION_H_
#define APPLICATION_H_

#include "gl_globals.hpp"
#include "cuda_globals.hpp"
#include "shader.hpp"
#include "image.hpp"
#include "fractal.hpp"

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
  HDD2,
  _COUNT
};


/// User Setting
struct UserSetting
{
  bool vsync = false;
  bool animation=false;
  double timeScale = 0.15;

  std::string outputDir="output";
  std::string prefix="img_";

  Resolution resolution = Resolution::_COUNT; // initialize with change_resolution
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
  void init_cuda();
  void cleanup();
  void create_shader();
  void create_gl_buffers();
  void run_cuda();
  void compute_fps();
  void recompute();

  void reset_gui();
  void gui_popcorn();
  void gui_mccabe();
  void gui_main();
  void change_resolution(Resolution state);

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
  Fractal::Fractals<T> fractals_;
  Fractal::Set fractal_later_;
  UserSetting settings_;

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
  unsigned texture_width_ = 0;
  unsigned texture_height_ = 0;
  unsigned texture_width_later_ = 0;
  unsigned texture_height_later_ = 0;

  double fps_ = 0.0;
  float ms_kernel_ = 0.0f;
  unsigned iteration_offset_ = 0;

  nanogui::ref<nanogui::Window> window_ = nullptr;
  nanogui::ref<nanogui::Window> window_params_ = nullptr;
  nanogui::ref<nanogui::Window> window_shading_ = nullptr;
  nanogui::Button* btnAnimate_ = nullptr;
  std::array<nanogui::FloatBox<double>*, 4> gui_coef_ = {{nullptr}};
  std::array<nanogui::FloatBox<double>*, 4> gui_zoom_ = {{nullptr}};
  nanogui::FloatBox<double>* gui_talpha_ = nullptr;
  nanogui::FloatBox<double>* gui_add_ = nullptr;
  nanogui::IntBox<unsigned>* gui_texWidth_ = nullptr;
  nanogui::IntBox<unsigned>* gui_texHeight_ = nullptr;
  nanogui::Slider* gui_iterations_ = nullptr;
};



#endif /* APPLICATION_H_ */

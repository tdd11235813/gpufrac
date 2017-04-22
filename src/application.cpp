#include "application.hpp"

#include <Eigen/Core>

#include <cuda_gl_interop.h>
#include <thread>
#include <iomanip>

template<typename T>
Application<T>::Application()
  : nanogui::Screen(Eigen::Vector2i(1024, 768),
                    "Fractal Dynamic Systems",
                    /*resizable*/true, /*fullscreen*/true, /*colorBits*/8,
                    /*alphaBits*/8, /*depthBits*/24, /*stencilBits*/8,
                    /*nSamples*/0, /*glMajor*/4, /*glMinor*/1
    )
{
  // start GLEW extension handler
  glewInit();

  gui_main();
  gui_popcorn();

  // initial values
  change_resolution(Resolution::SMALL);

  texture_width_ = texture_width_later_;
  texture_height_ = texture_height_later_;

  fractals_.current_ = fractal_later_ = Fractal::Set::POPCORN;

  create_gl_buffers();
  create_shader();
  create_quad();

  // --

  performLayout();

  resizeEvent(Eigen::Vector2i(0,0)); // initial window positions

  // FPS thread
  std::thread([&]()
              {
                while (1)
                {
                  std::ostringstream ss;
                  ss << std::setprecision(1) << std::fixed << fps_;
                  if(fractals_.current_ == Fractal::Set::POPCORN)
                    ss << " (t=" << std::fixed << std::setprecision(3)
                       << fractals_.popcorn_.params_.time << ")";
                  labelTime_->setCaption( ss.str() );
                  if(settings_.animation) {
                    ss.str("");
                    ss.clear();
                    ss << "Kernel = " << std::fixed << std::setprecision(1) << ms_kernel_ <<" ms";
                    labelStatus_->setCaption(ss.str());
                  }
                  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                  if(app_started_==false) {
                    recompute();
                    app_started_ = true;
                  }
                }
              }).detach();

  std::cout << "> Initialization done." << std::endl;
}

template<typename T>
Application<T>::~Application()
{
  std::cout << "> Teardown." << std::endl;
  cleanup();
}

template<typename T>
void Application<T>::cleanup()
{
  CHECK_CUDA(cudaStreamSynchronize(0));
  if(resource_) {
    CHECK_CUDA(cudaGraphicsUnregisterResource(resource_));
    resource_ = 0;
  }

  fractals_.cleanup();

  if(imagePBO_){
    glDeleteBuffers(1, &imagePBO_);
    imagePBO_=0;
  }
  if(texId_) {
    glDeleteTextures(1, &texId_);
    texId_=0;
  }
}

template<typename T>
void Application<T>::recompute() {
  recompute_ = true;
  if(fractals_.current_ == Fractal::Set::MCCABE)
    fractals_.mccabe_.init_buffer(false);
  else
    iteration_offset_ = 0;
}

template<typename T>
void Application<T>::run_cuda()
{
  switch(fractals_.current_) {
  case Fractal::Set::MCCABE:
    ms_kernel_ = fractals_.mccabe_.launch_kernel(resource_, settings_.animation);
    break;
  default:
    ms_kernel_ = fractals_.popcorn_.launch_kernel(resource_, iteration_offset_);
  }
}

template<typename T>
void Application<T>::draw(NVGcontext* ctx)
{
  compute_fps();

  if(fractals_.width() != texture_width_
     || fractals_.height() != texture_height_
     || texture_width_later_ != texture_width_
     || texture_height_later_ != texture_height_) {
    rescale_ = true;
  }

  if(fractal_later_ != fractals_.current_) {
    this->change_resolution(Resolution::SMALL);
    fractals_.current_ = fractal_later_;
    reset_gui();
    rescale_ = true;
  }

  if(rescale_) {
    cleanup();
    texture_width_ = texture_width_later_;
    texture_height_ = texture_height_later_;

    gui_texWidth_->setValue(texture_width_);
    gui_texHeight_->setValue(texture_height_);

    fractals_.set_size(texture_width_, texture_height_);
    create_gl_buffers();
    init_cuda();
    rescale_ = false;
    recompute();
  }else if(reset_buffer_){

    switch(fractals_.current_) {
    case Fractal::Set::MCCABE:
      fractals_.mccabe_.init_buffer(false);
      break;
    default:
      fractals_.popcorn_.init_buffer();
    }
    reset_buffer_=false;
  }

  if(screenshot_){
    try{
      auto s = TGAImage::savePBO(imagePBO_,
                                 texture_width_,
                                 texture_height_,
                                 settings_.outputDir,
                                 settings_.prefix);
      labelStatus_->setCaption(std::string("Saved to: ") + s);
    }catch(const std::runtime_error& e){
      using namespace nanogui;
      new MessageDialog(
        this,
        MessageDialog::Type::Warning,
        "Could not save image!",
        std::string("Error: ")+e.what());
    }
    screenshot_ = false;
  }

  // compute part

  static double oldtime = glfwGetTime();
  double current_time = glfwGetTime();
  if (settings_.animation == true || recompute_)
  {
    if(settings_.animation) {
      double delta = settings_.timeScale * (current_time - oldtime);
      switch(fractals_.current_) {
      case Fractal::Set::POPCORN:
        iteration_offset_ = 0;
        fractals_.popcorn_.params_.time_delta = delta;
        fractals_.popcorn_.params_.time += delta;
        break;
      case Fractal::Set::MCCABE:
        fractals_.mccabe_.params_.time_delta = delta;
        fractals_.mccabe_.params_.time += delta;
        break;
      }
      run_cuda();
    }else{ // animation is false, but recompute is true
      run_cuda();
      if(fractals_.current_ == Fractal::Set::POPCORN) {
        iteration_offset_ += fractals_.popcorn_.params_.iterations_per_run;
        if( iteration_offset_ >= fractals_.popcorn_.params_.iterations) {
          iteration_offset_ = 0;
          labelStatus_->setCaption("Finished!");
        }else
          labelStatus_->setCaption("Rendering ...");
      }
      recompute_ = false;
    }
  }
  oldtime = current_time;

  /* Draw the user interface */
  Screen::draw(ctx);
}

template<typename T>
void Application<T>::drawContents()
{
  unsigned width = texture_width_;
  unsigned height = texture_height_;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imagePBO_);
  glBindTexture(GL_TEXTURE_2D, texId_);

  // Note: glTexSubImage2D will perform a format conversion if the
  // buffer is a different format from the texture. We created the
  // texture with format GL_RGBA8. In glTexSubImage2D we specified
  // GL_BGRA and GL_UNSIGNED_BYTE. This is a fast-path combination

  // Note: NULL indicates the data resides in device memory
  // hence data is coming from PBO
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                  GL_RGBA, GL_UNSIGNED_BYTE, NULL);

  shader_.bind();
  glBindVertexArray(vertexArrayQuad_);
  glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, 0);
  glBindVertexArray(0);

  shader_.unbind();
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  if (CHECK_GLERROR() != GL_NO_ERROR)
    throw std::runtime_error("OpenGL error while drawing.");
}

template<typename T>
void Application<T>::init_cuda()
{
  fractals_.set_size(texture_width_, texture_height_);
  switch(fractals_.current_) {
  case Fractal::Set::MCCABE:
    fractals_.mccabe_.init_buffer(true);
    break;
  default:
    fractals_.popcorn_.alloc_buffer();
    fractals_.popcorn_.init_buffer();
  }
}

template<typename T>
void Application<T>::create_shader()
{
  if (shader_.isLoaded())
    shader_.unload();
  shader_.load("shaders/quad.vert", "shaders/quad.frag");
  shader_.link();

  shader_.bind();
  shader_.setUniformVar("image",0);
  shader_.unbind();
}

template<typename T>
bool Application<T>::keyboardEvent(int key, int scancode, int action, int modifiers)
{
  if (Screen::keyboardEvent(key, scancode, action, modifiers))
    return true;
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
  {
    setVisible(false);
    return true;
  }

  if (key == GLFW_KEY_A && action == GLFW_PRESS)
  {
    settings_.animation = !settings_.animation;
    btnAnimate_->setPushed(settings_.animation);
    labelStatus_->setCaption("");
    if(settings_.animation==false)
      recompute();
    return true;
  }

  if (key == GLFW_KEY_M && action == GLFW_PRESS)
  {
    window_->setVisible(!window_->visible());
    window_shading_->setVisible(!window_shading_->visible());
    window_params_->setVisible(!window_params_->visible());
    return true;
  }

  if (key == GLFW_KEY_S && action == GLFW_PRESS)
  {
    screenshot_=true;
    return true;
  }

  if (key == GLFW_KEY_S && action == GLFW_PRESS)
  {
    screenshot_=true;
    return true;
  }
  return false;
}

template<typename T>
void Application<T>::create_quad()
{
  struct VData {
    float vertex[4];
    float texcoord[2];
    float normal[3];
  } quad[4] = {{
      {-1.f,-1.f, 0.f, 1.0f}, //vertex
      {0.f, 0.f},            //texcoord
      {0.f, 0.f, 1.0f}       //normal
    },{
      {-1.f, 1.f, 0.f, 1.0f},
      {0.f, 1.f},
      {0.f, 0.f, 1.0f}
    },{
      {1.f, 1.f, 0.f, 1.0f},
      {1.f, 1.f},
      {0.f, 0.f, 1.0f}
    },{
      {1.f,-1.f, 0.f, 1.0f},
      {1.f, 0.f},
      {0.f, 0.f, 1.0f}
    }
  };

  if(!quadFinalVBOId_)
    glGenBuffers(1,&quadFinalVBOId_); // on screen quad for rendering
  if(!quadIBOId_)
    glGenBuffers(1,&quadIBOId_);
  if(!vertexArrayQuad_)
    glGenVertexArrays(1, &vertexArrayQuad_);

  GLushort quadInd[] = {
    0,1,3,2
  };

  upload( quadFinalVBOId_, (char*)quad,
          4*sizeof(struct VData),
          GL_ARRAY_BUFFER, GL_STATIC_DRAW );

  upload( quadIBOId_, (char*)quadInd,
          4*sizeof(GLushort),
          GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW );

  //bind vertex array for quad
  glBindVertexArray(vertexArrayQuad_);
  glBindBuffer(GL_ARRAY_BUFFER, quadFinalVBOId_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadIBOId_);

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glEnableVertexAttribArray(2);

  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 9*sizeof(GLfloat), 0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 9*sizeof(GLfloat), (GLvoid*)(4*sizeof(GLfloat)));
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9*sizeof(GLfloat), (GLvoid*)(6*sizeof(GLfloat)));
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

template<typename T>
void Application<T>::create_gl_buffers()
{
  if(!texId_)
    glGenTextures(1,&texId_);

  unsigned width = texture_width_;
  unsigned height = texture_height_;
  unsigned total = 4*width*height;
  GLubyte* h = new GLubyte[total];
  // initialization, text image
  for(unsigned k=0; k<total; k+=4){
    h[k+0] = 0;
    h[k+1] = 0;
    h[k+2] = 0;
    h[k+3] = 255;
  }
  glBindTexture(GL_TEXTURE_2D, texId_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);

  glGenBuffers(1, &imagePBO_);

  upload(imagePBO_, (char*)h, total*sizeof(GLubyte), GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW);

  CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&resource_,imagePBO_,cudaGraphicsRegisterFlagsWriteDiscard));

  delete[] h;
  if (CHECK_GLERROR() != GL_NO_ERROR)
    throw std::runtime_error("Could not create OpenGL buffers.");
}

template<typename T>
void Application<T>::compute_fps()
{
  static double lastTime = 0.0;
  static size_t nbFrames = 0;
  nbFrames++;
  double currentTime = glfwGetTime();
  double delta = currentTime - lastTime;
  if ( delta >= 1.0 ){
    fps_ = double(nbFrames) / delta;
    nbFrames = 0;
    lastTime = currentTime;
  }
}

template<typename T>
bool Application<T>::mouseMotionEvent(const Eigen::Vector2i& p,
                                      const Eigen::Vector2i& rel, int btn, int mdf)
{
  Screen::mouseMotionEvent(p, rel, btn, mdf);
  if(fractals_.current_ != Fractal::Set::MCCABE && mousePressed_) {
    auto& params = fractals_.popcorn_.params_;
    double dx = (params.x1-params.x0)*(1.0*rel.x()/width()); // assuming fullscreen
    double dy = -(params.y1-params.y0)*(1.0*rel.y()/height());
    params.x0 -= dx;
    params.x1 -= dx;
    params.y0 -= dy;
    params.y1 -= dy;
    gui_x0_->setValue(params.x0);
    gui_x1_->setValue(params.x1);
    gui_y0_->setValue(params.y0);
    gui_y1_->setValue(params.y1);
    recompute();
    return true;
  }
  return false;
}

template<typename T>
bool Application<T>::mouseButtonEvent(const Eigen::Vector2i& p, int button, bool down,
                                      int modifiers)
{
  Screen::mouseButtonEvent(p, button, down, modifiers);
  if(down && button == GLFW_MOUSE_BUTTON_LEFT
     && (window_->visible()==false || window_->contains(p)==false)
     && (window_shading_->visible()==false || window_shading_->contains(p)==false)
     && (window_params_->visible()==false || window_params_->contains(p)==false)
    ){
    mousePressed_ = true;
    setCursor(nanogui::Cursor::Crosshair);
  }else{
    mousePressed_ = false;
    setCursor(nanogui::Cursor::Arrow);
  }
  return true;
}

template<typename T>
bool Application<T>::scrollEvent(const Eigen::Vector2i& p,
                                 const Eigen::Vector2f& rel)
{
  Screen::scrollEvent(p, rel);
  if(fractals_.current_ != Fractal::Set::MCCABE
     && (window_->visible()==false || window_->contains(p)==false)
     && (window_shading_->visible()==false || window_shading_->contains(p)==false)
     && (window_params_->visible()==false || window_params_->contains(p)==false)
    ) {
    double vx = 1.0*p.x()/width();
    double vy = 1.0-1.0*p.y()/height();
    auto& params = fractals_.popcorn_.params_;

    double zfactor_x = 0.1*rel.y()*(params.x0-params.x1);
    double zfactor_y = 0.1*rel.y()*(params.y0-params.y1);
    params.x0 -= vx*zfactor_x;
    params.x1 += (1.0-vx)*zfactor_x;
    params.y0 -= vy*zfactor_y;
    params.y1 += (1.0-vy)*zfactor_y;

    gui_x0_->setValue(params.x0);
    gui_x1_->setValue(params.x1);
    gui_y0_->setValue(params.y0);
    gui_y1_->setValue(params.y1);
    recompute();
    return true;
  }
  return false;
}

template<typename T>
inline bool
Application<T>::resizeEvent(const Eigen::Vector2i& )
{
  window_->setPosition(Eigen::Vector2i(0, height()-window_->height()+(height()>500?-100:0)));
  window_params_->setPosition(Eigen::Vector2i((width()-window_params_->width())/3, height()-window_params_->height()));
  window_shading_->setPosition(Eigen::Vector2i((width()+window_shading_->width())/2, height()-window_shading_->height()));
  labelTime_->setWidth(260);
  labelStatus_->setWidth(260);
  return true;
}


template<typename T>
void Application<T>::gui_popcorn() {

  using namespace nanogui;
  FormHelper *gui = new FormHelper(this);

  if(window_params_)
    window_params_->dispose();
  window_params_ = gui->addWindow(Eigen::Vector2i(100, 10), "Popcorn");

  auto& params = fractals_.popcorn_.params_;

  gui->addGroup("Parameters");

  auto* cobo = gui->addVariable("Popcorns", fractals_.popcorn_.current_);
  cobo->setItems({"V1", "V2", "Classic", "Test"});
  cobo->setCallback([&](auto state) {
    if(state != fractals_.popcorn_.current_) {
      fractals_.popcorn_.current_ = state;
      this->recompute();
    }
  });


  IntBox<unsigned>* box = gui->addVariable("MaxIterations", params.max_iterations);
  box->setCallback([&](auto v){this->update_value(params.max_iterations, v); } );
  box->setValueIncrement(32);
  box->setSpinnable(true);
  box->setMinValue(1);
  box = gui->addVariable("Iterations", params.iterations);
  box->setCallback([&](auto v){this->update_value(params.iterations, v); } );
  box->setValueIncrement(16);
  box->setSpinnable(true);
  box->setMinValue(1);


  box = gui->addVariable("IterationsPerRun", params.iterations_per_run);
  box->setCallback([&](auto v){this->update_value(params.iterations_per_run, v); } );
  box->setValueIncrement(16);
  box->setSpinnable(true);
  box->setMinMaxValues(1, 128);

  std::array<std::string, 4> xylabels = {{"x0", "x1", "y0", "y1"}};
  gui_x0_ = gui->addVariable(xylabels[0], params.x0);
  gui_x0_->setValueIncrement(0.05);
  gui_x0_->setSpinnable(true);
  gui_x0_->setCallback([&](auto v){ this->update_value(params.x0, v); });
  gui_x1_ = gui->addVariable(xylabels[1], params.x1);
  gui_x1_->setValueIncrement(0.05);
  gui_x1_->setSpinnable(true);
  gui_x1_->setCallback([&](auto v){ this->update_value(params.x1, v); });
  gui_y0_ = gui->addVariable(xylabels[2], params.y0);
  gui_y0_->setValueIncrement(0.05);
  gui_y0_->setSpinnable(true);
  gui_y0_->setCallback([&](auto v){ this->update_value(params.y0, v); });
  gui_y1_ = gui->addVariable(xylabels[3], params.y1);
  gui_y1_->setValueIncrement(0.05);
  gui_y1_->setSpinnable(true);
  gui_y1_->setCallback([&](auto v){this->update_value(params.y1, v);});

  std::array<std::string, 4> cflabels = {{"c0", "c1", "c2", "c3"}};
  auto* coeffbox = gui->addVariable(cflabels[0], params.t0);
  coeffbox->setValueIncrement(0.05);
  coeffbox->setSpinnable(true);
  coeffbox->setCallback([&](auto v){ this->update_value(params.t0, v); });
  coeffbox = gui->addVariable(cflabels[1], params.t1);
  coeffbox->setValueIncrement(0.05);
  coeffbox->setSpinnable(true);
  coeffbox->setCallback([&](auto v){ this->update_value(params.t1, v); });
  coeffbox = gui->addVariable(cflabels[2], params.t2);
  coeffbox->setValueIncrement(0.05);
  coeffbox->setSpinnable(true);
  coeffbox->setCallback([&](auto v){ this->update_value(params.t2, v); });
  coeffbox = gui->addVariable(cflabels[3], params.t3);
  coeffbox->setValueIncrement(0.05);
  coeffbox->setSpinnable(true);
  coeffbox->setCallback([&](auto v){this->update_value(params.t3, v);});

  coeffbox = gui->addVariable("Lambda", params.talpha);
  coeffbox->setValueIncrement(0.005);
  coeffbox->setSpinnable(true);
  coeffbox->setCallback([&](auto v){this->update_value(params.talpha, v);});

  coeffbox = gui->addVariable("Hit value", params.hit_value);
  coeffbox->setValueIncrement(0.001);
  coeffbox->setMinValue(0.001);
  coeffbox->setSpinnable(true);
  coeffbox->setCallback([&](auto v){this->update_value(params.hit_value, v);});

  // -- Coloring --
  if(window_shading_)
    window_shading_->dispose();
  window_shading_ = gui->addWindow(Eigen::Vector2i(10, 10), "Rendering");
  window_shading_->setWidth(220);
  gui->addGroup("Colors");
  auto* cbox = gui->addVariable("HSL Mode", params.hslMode);
  cbox->setCallback([&](auto v){this->update_value(params.hslMode, v);} );

  cbox = gui->addVariable("SubSampling", params.sub_sampling);
  cbox->setCallback([&](auto v){this->update_value(params.sub_sampling, v);} );

  cbox = gui->addVariable("PixelTrace", params.pixel_trace);

  IntBox<unsigned>* ibox = gui->addVariable("Divisor", params.pixel_trace_divisor);
  ibox->setCallback([&](auto v){this->update_value(params.pixel_trace_divisor, v);} );
  ibox->setEnabled(false);
  ibox->setSpinnable(true);
  ibox->setMinValue(0);

  cbox->setCallback([&,ibox](auto v){this->update_value(params.pixel_trace, v); ibox->setEnabled(v);} );

  // Slider *slider = new Slider(gui->window());
  // slider->setValue(0.0f);
  // slider->setFixedWidth(80);
  // slider->setFinalCallback([&](float v){this->update_value(params.hue_start, v); });
  // gui->addWidget("Hue", slider);

  auto* vbox = gui->addVariable("HueStart", params.hue_start);
  vbox->setCallback([&](auto v){this->update_value(params.hue_start, v); });
  vbox->setSpinnable(true);
  vbox->setMinMaxValues(-1.0, 1.0);
  vbox->setValueIncrement(0.01);

  vbox = gui->addVariable("HueEnd", params.hue_end);
  vbox->setCallback([&](auto v){this->update_value(params.hue_end, v); });
  vbox->setSpinnable(true);
  vbox->setMinMaxValues(-1.0, 1.0);
  vbox->setValueIncrement(0.01);

  vbox = gui->addVariable("HueSlope", params.hue_slope);
  vbox->setCallback([&](auto v){this->update_value(params.hue_slope, v); });
  vbox->setSpinnable(true);
  vbox->setValueIncrement(0.05);

  vbox = gui->addVariable("DensitySlope", params.density_slope);
  vbox->setCallback([&](auto v){this->update_value(params.density_slope, v); });
  vbox->setSpinnable(true);
  vbox->setValueIncrement(0.05);

  vbox = gui->addVariable("SaturationSlope", params.saturation_slope);
  vbox->setCallback([&](auto v){this->update_value(params.saturation_slope, v); });
  vbox->setSpinnable(true);
  vbox->setValueIncrement(0.05);

  vbox = gui->addVariable("BrightnessSlope", params.brightness_slope);
  vbox->setCallback([&](auto v){this->update_value(params.brightness_slope, v); });
  vbox->setSpinnable(true);
  vbox->setValueIncrement(0.05);


  cbox = gui->addVariable("Invert", params.invert);
  cbox->setCallback([&](auto v){this->update_value(params.invert, v); });
  cbox = gui->addVariable("UseAtomics", params.use_atomics);
  cbox->setCallback([&](auto v){this->update_value(params.use_atomics, v); });

}

template<typename T>
void Application<T>::gui_mccabe() {

  using namespace nanogui;
  FormHelper *gui = new FormHelper(this);

  if(window_params_)
    window_params_->dispose();
  window_params_ = gui->addWindow(Eigen::Vector2i(100, 10), "Turing McCabe");

  auto& params = fractals_.mccabe_.params_;

  gui->addGroup("Parameters");
  auto* vbox = gui->addVariable("Base", params.base);
  vbox->setCallback([&](auto v){this->update_value(params.base, v); });
  vbox->setSpinnable(true);
  vbox->setValueIncrement(0.1);
  vbox->setMinValue(1.1);

  vbox = gui->addVariable("stepScale", params.stepScale);
  vbox->setCallback([&](auto v){this->update_value(params.stepScale, v); });
  vbox->setSpinnable(true);
  vbox->setValueIncrement(0.001);

  vbox = gui->addVariable("stepOffset", params.stepOffset);
  vbox->setCallback([&](auto v){this->update_value(params.stepOffset, v); });
  vbox->setSpinnable(true);
  vbox->setValueIncrement(0.001);

  vbox = gui->addVariable("blurFactor", params.blurFactor);
  vbox->setCallback([&](auto v){this->update_value(params.blurFactor, v); });
  vbox->setSpinnable(true);
  vbox->setValueIncrement(0.5);

  auto *ibox = gui->addVariable("Symmetry", params.symmetry);
  ibox->setCallback([&](auto v){this->update_value(params.symmetry, v); });
  ibox->setSpinnable(true);
  ibox->setMinMaxValues(0,4);

  ibox = gui->addVariable("DirectionMode", params.direction_mode);
  ibox->setCallback([&](auto v){this->update_value(params.direction_mode, v); });
  ibox->setSpinnable(true);
  ibox->setMinMaxValues(0,2);

  ibox = gui->addVariable("Seed", params.seed);
  ibox->setCallback([&](auto v){this->update_value(params.seed, v); });
  ibox->setSpinnable(true);
  ibox->setMinValue(0);


  // -- Coloring --
  if(window_shading_)
    window_shading_->dispose();
  window_shading_ = gui->addWindow(Eigen::Vector2i(10, 10), "Rendering");
  window_shading_->setWidth(220);
  gui->addGroup("Colors");

  // Slider *slider = new Slider(gui->window());
  // slider->setValue(0.0f);
  // slider->setFixedWidth(80);
  // slider->setFinalCallback([&](float v){this->update_value(params.hue_start, v); });
  // gui->addWidget("Hue", slider);

  vbox = gui->addVariable("HueStart", params.hue_start);
  vbox->setCallback([&](auto v){this->update_value(params.hue_start, v); });
  vbox->setSpinnable(true);
  vbox->setMinMaxValues(-1.0, 1.0);
  vbox->setValueIncrement(0.01);

  vbox = gui->addVariable("HueEnd", params.hue_end);
  vbox->setCallback([&](auto v){this->update_value(params.hue_end, v); });
  vbox->setSpinnable(true);
  vbox->setMinMaxValues(-1.0, 1.0);
  vbox->setValueIncrement(0.01);

  vbox = gui->addVariable("HueSlope", params.hue_slope);
  vbox->setCallback([&](auto v){this->update_value(params.hue_slope, v); });
  vbox->setSpinnable(true);
  vbox->setValueIncrement(0.05);

  vbox = gui->addVariable("DensitySlope", params.density_slope);
  vbox->setCallback([&](auto v){this->update_value(params.density_slope, v); });
  vbox->setSpinnable(true);
  vbox->setValueIncrement(0.05);

  vbox = gui->addVariable("SaturationSlope", params.saturation_slope);
  vbox->setCallback([&](auto v){this->update_value(params.saturation_slope, v); });
  vbox->setSpinnable(true);
  vbox->setValueIncrement(0.05);

  vbox = gui->addVariable("BrightnessSlope", params.brightness_slope);
  vbox->setCallback([&](auto v){this->update_value(params.brightness_slope, v); });
  vbox->setSpinnable(true);
  vbox->setValueIncrement(0.05);

  auto* cbox = gui->addVariable("Invert", params.invert);
  cbox->setCallback([&](auto v){this->update_value(params.invert, v); });

}

template<typename T>
void Application<T>::gui_main() {
  using namespace nanogui;
  FormHelper *gui = new FormHelper(this);

  // --- Properties ---

  window_ = gui->addWindow(Eigen::Vector2i(10, 10), "gpufrac");

  // -- Control --

  gui->addGroup("Control");
  auto* fbox = gui->addVariable("Time scale", settings_.timeScale);
  fbox->setSpinnable(true);
  fbox->setValueIncrement(0.05);

  auto* cobo = gui->addVariable("FractalType", fractal_later_);
  cobo->setItems({"Popcorn", "Turing McCabe"});
  // cobo->setCallback([&](auto state) {
  //   if(state != fractal_later_) {
  //     fractal_later_ = state;
  //   }
  // });

  gui->addButton("Reset Time", [&](){fractals_.reset_time(); recompute();});
  gui->addButton("Reset Fractal", [&](){
                                    fractals_.reset_current();
                                    reset_gui();
                                  });


  // -- Image --
  gui->addGroup("Image");
  gui_texWidth_ = gui->addVariable("Width", texture_width_later_);
  gui_texWidth_->setMinMaxValues(32, 4096);
  gui_texHeight_ = gui->addVariable("Height", texture_height_later_);
  gui_texHeight_->setMinMaxValues(32, 4096);

  auto* cobores = gui->addVariable("Resolution", settings_.resolution);
  cobores->setItems({"S", "M", "SD", "720p", "1080p", "2160p"});
  cobores->setCallback([&](Resolution state) {
    change_resolution(state);
  });

  gui->addVariable("ImgOutputDir", settings_.outputDir);
  gui->addVariable("ImgPrefix", settings_.prefix);

  gui->addButton("Screenshot [s]", [&](){screenshot_=true;});
  Button* button = btnAnimate_ = gui->addButton("Animate [a]", nullptr);
  button->setFlags(Button::ToggleButton);
  button->setChangeCallback([&](bool state){
    settings_.animation=state;
    if(state==false)
      this->recompute();
    labelStatus_->setCaption("");
  });

  labelTime_ = new Label(window_, "");
  labelTime_->setColor(nanogui::Color(150,255));

  labelStatus_ = new Label(window_, "");
  labelStatus_->setColor(nanogui::Color(30,205,30,255));

  gui->addWidget("FPS: ", labelTime_);
  gui->addWidget("Status: ", labelStatus_);
}

template<typename T>
void Application<T>::change_resolution(Resolution state) {
  if(state!=settings_.resolution)
  {
    switch(state)
    {
    case Resolution::SMALL: texture_height_later_ = texture_width_later_ = 400; break;
    case Resolution::MEDIUM: texture_height_later_ = texture_width_later_ = 800; break;
    case Resolution::SD: texture_height_later_ = 576; texture_width_later_ = 720; break;
    case Resolution::HD: texture_height_later_ = 720; texture_width_later_ = 1280; break;
    case Resolution::HD2: texture_height_later_ = 1080; texture_width_later_ = 1920; break;
    case Resolution::HDD2: texture_height_later_ = 2160; texture_width_later_ = 3840; break;
    }
    settings_.resolution = state;
    rescale_ = true;
  }
}

template<typename T>
void Application<T>::reset_gui() {

  if(fractals_.current_ == Fractal::Set::MCCABE) {
    this->gui_mccabe();
  } else {
    this->gui_popcorn();
  }
  performLayout();
  resizeEvent(Eigen::Vector2i(0,0));
}


template class Application<float>;

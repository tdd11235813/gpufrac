/*
 * application.cpp
 *
 *  Created on: 17.09.2016
 *      Author: mwerner
 */

#include "application.h"

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
  cout << "> Create OpenGL Buffers." << endl;
  create_buffers();
  cout << "> Init Shaders." << endl;
  create_shader();
  create_quad();
  cout << "> Init Cuda Buffers." << endl;
  initCuda(); // init data on gpu

  using namespace nanogui;

  FormHelper *gui = new FormHelper(this);
  window_ = gui->addWindow(Eigen::Vector2i(10, 10), "Properties");

  gui->addGroup("Parameters");
  IntBox<unsigned>* box = gui->addVariable("Iterations", parameters_.max_iterations);
  box->setCallback([&](auto v){
    mutex_max_iterations_.lock();
    parameters_.max_iterations = v;
    tmp_max_iterations_ = v;
    mutex_max_iterations_.unlock();
    redraw_ = true;
  } );
  box->setValueIncrement(16);
  box->setSpinnable(true);
  box->setMinMaxValues(1, 1024);
  tmp_max_iterations_ = parameters_.max_iterations;

  std::array<std::string, 4> cflabels = {{"c0", "c1", "c2", "c3"}};
  auto* coeffbox = gui->addVariable(cflabels[0], parameters_.t0);
  coeffbox->setValueIncrement(0.05);
  coeffbox->setSpinnable(true);
  coeffbox->setCallback([&](auto v){parameters_.t0 = v; redraw_=true;});
  coeffbox = gui->addVariable(cflabels[1], parameters_.t1);
  coeffbox->setValueIncrement(0.05);
  coeffbox->setSpinnable(true);
  coeffbox->setCallback([&](auto v){parameters_.t1 = v; redraw_=true;});
  coeffbox = gui->addVariable(cflabels[2], parameters_.t2);
  coeffbox->setValueIncrement(0.05);
  coeffbox->setSpinnable(true);
  coeffbox->setCallback([&](auto v){parameters_.t2 = v; redraw_=true;});
  coeffbox = gui->addVariable(cflabels[3], parameters_.t3);
  coeffbox->setValueIncrement(0.05);
  coeffbox->setSpinnable(true);
  coeffbox->setCallback([&](auto v){parameters_.t3 = v; redraw_=true;});

  coeffbox = gui->addVariable("Lambda", parameters_.talpha);
  coeffbox->setValueIncrement(0.005);
  coeffbox->setSpinnable(true);
  coeffbox->setCallback([&](auto v){parameters_.talpha = v; redraw_=true;});

  coeffbox = gui->addVariable("Hit value", parameters_.addValue);
  coeffbox->setValueIncrement(0.001);
  coeffbox->setMinValue(0.001);
  coeffbox->setSpinnable(true);
  coeffbox->setCallback([&](auto v){parameters_.addValue = v; redraw_=true;});


  gui->addGroup("Image");
  gui_texWidth_ = gui->addVariable("texWidth", parameters_.width);
  gui_texWidth_->setCallback([&](auto v){tmp_texWidth_ = v; rescale_ = true;} );
  gui_texWidth_->setMinMaxValues(32, 2048);
  gui_texHeight_ = gui->addVariable("texHeight", parameters_.height);
  gui_texHeight_->setCallback([&](auto v){tmp_texHeight_ = v; rescale_ = true;} );
  gui_texHeight_->setMinMaxValues(32, 2048);
  tmp_texWidth_ = parameters_.width;
  tmp_texHeight_ = parameters_.height;

  auto* cobores = gui->addVariable("Resolution", resolution_);
  cobores->setItems({"S", "M", "SD", "720p", "1080p"});
  cobores->setCallback([&](Resolution state) {
    if(state!=resolution_)
    {
      switch(state)
      {
        case Resolution::SMALL: tmp_texHeight_ = tmp_texWidth_ = 400; break;
        case Resolution::MEDIUM: tmp_texHeight_ = tmp_texWidth_ = 800; break;
        case Resolution::SD: tmp_texHeight_ = 576; tmp_texWidth_ = 720; break;
        case Resolution::HD: tmp_texHeight_ = 720; tmp_texWidth_ = 1280; break;
        case Resolution::HD2: tmp_texHeight_ = 1080; tmp_texWidth_ = 1920; break;
      }
      gui_texWidth_->setValue(tmp_texWidth_);
      gui_texHeight_->setValue(tmp_texHeight_);
      resolution_ = state;
      rescale_ = true;
    }
  });
  gui->addVariable("ImgOutputDir", settings_.outputDir);
  gui->addVariable("ImgPrefix", settings_.prefix);

  gui->addGroup("Coloring");
  auto* hsbbox = gui->addVariable("HSL Mode", settings_.hslMode);
  hsbbox->setCallback([&](auto v){settings_.hslMode = v; redraw_ = true;} );

  Slider *slider = new Slider(gui->window());
  slider->setValue(0.0f);
  slider->setFixedWidth(80);
  slider->setFinalCallback([&](float value)
  {
    parameters_.hueOffset = value;
    redraw_ = true;
  });
  gui->addWidget("Hue", slider);


  gui->addGroup("Control");
  auto* fbox = gui->addVariable("Time scale", settings_.timeScale);
  fbox->setSpinnable(true);
  fbox->setValueIncrement(0.05);

  auto* cobo = gui->addVariable("Fractal", settings_.fractal);
  cobo->setItems({"Popcorn 1", "Popcorn 2", "Popcorn 3", "Popcorn 4"});
  cobo->setCallback([&](auto state) { settings_.fractal = state; redraw_=true; });

  gui->addButton("Screenshot", [&](){screenshot_=true;});
  Button* button = btnAnimate_ = gui->addButton("Animate", nullptr);
  button->setFlags(Button::ToggleButton);
  button->setChangeCallback([&](bool state){settings_.animation=state;labelStatus_->setCaption("");});

  // status window
  window_status_ = new Window(this, "Status");
  window_status_->setLayout(new BoxLayout(Orientation::Vertical,
                                         Alignment::Minimum, 4, 6));
  (new Label(window_status_, "Keys: a .. animate | m .. menu", "sans-bold", 16));
  labelPosition_ = new Label(window_status_, " ", "sans-bold", 16);
  labelPosition_->setColor(nanogui::Color(150,255));
  labelStatus_ = new Label(window_status_, " ", "sans-bold", 16);
  labelStatus_->setColor(nanogui::Color(30,205,30,255));
  window_status_->setFixedSize(Eigen::Vector2i(240,90));

  performLayout();

  window_->setPosition(Vector2i(0, height()-window_->height()));
  window_status_->setPosition(Vector2i(width()-window_status_->width(), height()-window_status_->height()));

  // FPS thread
  std::thread([&]()
  {
    while (1)
    {
      std::ostringstream ss;
      ss << "Time: " << std::fixed << std::setprecision(3) << parameters_.time
         << std::setprecision(1) << "    [" << fps_ << " FPS]";
      window_status_->setTitle( ss.str() );
      ss.str("");
      ss.clear();
      ss << "(x0, x1): " << std::fixed << std::setprecision(2)
         << parameters_.x0 << ", " << parameters_.x1
         << " (y0, y1): "
         << parameters_.y0 << ", " << parameters_.y1;
      labelPosition_->setCaption(ss.str());
      if(settings_.animation) {
        ss.str("");
        ss.clear();
        ss << "Kernel: " << std::fixed << std::setprecision(1) << ms_kernel_ <<" ms";
        labelStatus_->setCaption(ss.str());
      }
      mutex_max_iterations_.lock();
      parameters_.max_iterations = tmp_max_iterations_;
      mutex_max_iterations_.unlock();
      redraw_ = true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }).detach();

  cout << "> Initialization done." << endl;
}

template<typename T>
Application<T>::~Application()
{
  cout << "> Teardown." << endl;
  cleanup();
}

template<typename T>
void Application<T>::cleanup()
{
  if(ddata_.buffer) {
    CHECK_CUDA(cudaStreamSynchronize(0));
    CHECK_CUDA(cudaGraphicsUnregisterResource(resource_));
    cleanup_cuda(ddata_);
    glDeleteBuffers(1, &imagePBO_);
    imagePBO_=0;
    glDeleteTextures(1, &texId_);
    texId_=0;
  }
}

template<typename T>
void Application<T>::runCuda()
{
  init_buffer(ddata_, parameters_);
  switch(settings_.fractal) {
    case Fractal::POPCORN0:
      if(settings_.hslMode)
        ms_kernel_ = launch_kernel<0,true>(resource_, ddata_, parameters_);
      else
        ms_kernel_ = launch_kernel<0,false>(resource_, ddata_, parameters_);
      break;
    case Fractal::POPCORN1:
      if(settings_.hslMode)
        ms_kernel_ = launch_kernel<1,true>(resource_, ddata_, parameters_);
      else
        ms_kernel_ = launch_kernel<1,false>(resource_, ddata_, parameters_);
      break;
    case Fractal::POPCORN2:
      if(settings_.hslMode)
        ms_kernel_ = launch_kernel<2,true>(resource_, ddata_, parameters_);
      else
        ms_kernel_ = launch_kernel<2,false>(resource_, ddata_, parameters_);
      break;
    case Fractal::POPCORN3:
      if(settings_.hslMode)
        ms_kernel_ = launch_kernel<3,true>(resource_, ddata_, parameters_);
      else
        ms_kernel_ = launch_kernel<3,false>(resource_, ddata_, parameters_);
      break;
  }
}

template<typename T>
void Application<T>::draw(NVGcontext* ctx)
{
  computeFPS();

  if(rescale_) {
    cleanup();
    parameters_.width = tmp_texWidth_;
    parameters_.height = tmp_texHeight_;
    create_buffers();
    initCuda();
    rescale_ = false;
    redraw_ = true;
  }else if(reset_buffer_){
    init_buffer(ddata_, parameters_);
    reset_buffer_=false;
  }

  if(screenshot_){
    try{
      auto s = TGAImage::saveOpenGLBuffer(parameters_.width,
                                          parameters_.height,
                                          settings_.outputDir,
                                          settings_.prefix);
      labelStatus_->setCaption(std::string("Saved to: ") + s);
    }catch(const std::runtime_error& e){
      using namespace nanogui;
      auto dlg = new MessageDialog(
          this,
          MessageDialog::Type::Warning,
          "Could not save image!",
          std::string("Error: ")+e.what());
    }
    screenshot_ = false;
  }

  static double oldtime = glfwGetTime();
  double current_time = glfwGetTime();
  if (settings_.animation == true || redraw_)
  {
    double delta = settings_.timeScale * (current_time - oldtime);
    if(settings_.animation) {
      unsigned tmpit = parameters_.max_iterations;
      parameters_.time += delta;
      parameters_.max_iterations = tmpit>128 ? 128 : tmpit;
      runCuda();
      parameters_.max_iterations = tmpit;
    }else
      runCuda();
    redraw_ = false;
  }
  oldtime = current_time;

  /* Draw the user interface */
  Screen::draw(ctx);
}

template<typename T>
void Application<T>::drawContents()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imagePBO_);
  glBindTexture(GL_TEXTURE_2D, texId_);

  // Note: glTexSubImage2D will perform a format conversion if the
  // buffer is a different format from the texture. We created the
  // texture with format GL_RGBA8. In glTexSubImage2D we specified
  // GL_BGRA and GL_UNSIGNED_BYTE. This is a fast-path combination

  // Note: NULL indicates the data resides in device memory
  // hence data is coming from PBO
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, parameters_.width, parameters_.height,
      GL_RGBA, GL_UNSIGNED_BYTE, NULL);

  shader_.bind();
  shader_.setUniformVar("image",0);
  glBindVertexArray(vertexArrayQuad_);
  glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, 0);
  glBindVertexArray(0);

  shader_.unbind();
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  if (CHECK_GLERROR() != GL_NO_ERROR)
    throw runtime_error("OpenGL error while drawing.");
}

template<typename T>
void Application<T>::initCuda()
{
  parameters_.n = parameters_.width*parameters_.height;
  alloc_buffer(ddata_, parameters_);
  init_buffer(ddata_, parameters_);
}

template<typename T>
void Application<T>::create_shader()
{
  if (shader_.isLoaded())
    shader_.unload();
  shader_.load("shaders/quad.vert", "shaders/quad.frag");
  shader_.link();
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
    return true;
  }
  if (key == GLFW_KEY_M && action == GLFW_PRESS)
  {
    window_->setVisible(!window_->visible());
    window_status_->setVisible(!window_status_->visible());
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
void Application<T>::create_buffers()
{
  if(!texId_)
    glGenTextures(1,&texId_);

  unsigned texwidth = parameters_.width;
  unsigned texheight= parameters_.height;
  unsigned total = 4*texwidth*texheight;
  GLubyte* h = new GLubyte[total];
  // initialization, text image
  for(unsigned k=0; k<total; k+=4){
    h[k+0] = 0;
    h[k+1] = 0;
    h[k+2] = 0;
    h[k+3] = 255;
  }
  glBindTexture(GL_TEXTURE_2D, texId_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texwidth, texheight, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);

  glGenBuffers(1, &imagePBO_);

  upload(imagePBO_, (char*)h, total*sizeof(GLubyte), GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_DRAW);

  CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&resource_,imagePBO_,cudaGraphicsRegisterFlagsWriteDiscard));

  delete[] h;
  if (CHECK_GLERROR() != GL_NO_ERROR)
    throw runtime_error("Could not create OpenGL buffers.");
}

template<typename T>
void Application<T>::computeFPS()
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
  if(mousePressed_) {
    double dx = (parameters_.x1-parameters_.x0)*(1.0*rel.x()/width()); // assuming fullscreen
    double dy = -(parameters_.y1-parameters_.y0)*(1.0*rel.y()/height());
    parameters_.x0 -= dx;
    parameters_.x1 -= dx;
    parameters_.y0 -= dy;
    parameters_.y1 -= dy;
    mutex_max_iterations_.lock();
    parameters_.max_iterations = parameters_.max_iterations<64?parameters_.max_iterations:64;
    mutex_max_iterations_.unlock();
    redraw_ = true;
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
      && (window_status_->visible()==false || window_status_->contains(p)==false)){
    mousePressed_ = true;
    setCursor(nanogui::Cursor::Crosshair);
  }else{
    mousePressed_ = false;
    setCursor(nanogui::Cursor::Arrow);
  }
  return true;
}

template<typename T>
bool Application<T>::scrollEvent(const Eigen::Vector2i& p, const Eigen::Vector2f& rel)
{
  Screen::scrollEvent(p, rel);
  if(window_->contains(p)==false && window_status_->contains(p)==false) {
    double vx = 1.0*p.x()/width();
    double vy = 1.0-1.0*p.y()/height();

    double zfactor_x = 0.1*rel.y()*(parameters_.x1-parameters_.x0);
    double zfactor_y = 0.1*rel.y()*(parameters_.y1-parameters_.y0);
    parameters_.x0 -= vx*zfactor_x;
    parameters_.x1 += (1.0-vx)*zfactor_x;
    parameters_.y0 -= vy*zfactor_y;
    parameters_.y1 += (1.0-vy)*zfactor_y;
    mutex_max_iterations_.lock();
    parameters_.max_iterations = parameters_.max_iterations<64?parameters_.max_iterations:64;
    mutex_max_iterations_.unlock();
    redraw_ = true;
    return true;
  }
  return false;
}

template<typename T>
inline bool
Application<T>::resizeEvent(const Eigen::Vector2i& )
{
  window_->setPosition(Eigen::Vector2i(0, height()-window_->height()));
  window_status_->setPosition(Eigen::Vector2i(width()-window_status_->width(), height()-window_status_->height()));
  return true;
}

template class Application<float>;
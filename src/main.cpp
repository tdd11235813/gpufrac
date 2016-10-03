#include "application.h"
#include <nanogui/nanogui.h>

#include <iostream>


int main(int argc, char** argv)
{
  try
  {
    std::cout << listCudaDevices().str();
    CHECK_CUDA( cudaSetDevice(0) );

    nanogui::init();
    {
      nanogui::ref < Application<float> > app = new Application<float>();
      //nanogui::ref < Application<double> > app = new Application<double>();
      app->drawAll();
      app->setVisible(true);
      nanogui::mainloop();
    }
    nanogui::shutdown();
  }
  catch (const std::runtime_error &e)
  {
    std::string error_msg = std::string("Caught a fatal error: ")
        + std::string(e.what());
#if defined(_WIN32)
    MessageBoxA(nullptr, error_msg.c_str(), NULL, MB_ICONERROR | MB_OK);
#else
    std::cerr << error_msg << std::endl;
#endif
    throw e;
  }
  CHECK_CUDA(cudaDeviceReset());
  return EXIT_SUCCESS;
}

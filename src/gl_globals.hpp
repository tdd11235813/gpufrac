/*****************************************************************************/
/**
 * @file
 * @brief Some globals.
 * @author Matthias Werner
 * @sa http://11235813tdd.blogspot.de/
 *****************************************************************************/

#ifndef GL_GLOBALS_H_
#define GL_GLOBALS_H_

//-----------------------------------------------------------------------------

/*#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <Windows.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/freeglut.h>
#include <GL/wglext.h>
#include <time.h>
typedef unsigned int uint;
#define M_WINDOWS 1
#endif

#if defined(LINUX) || defined(__linux)
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/freeglut.h>
#include <GL/glx.h>
#include <string.h>
#include <sys/time.h>
#endif*/
#include <GL/glew.h>
#include <nanogui/opengl.h>

#include <iostream>
#include <stdexcept>
// define this to disable OpenGL error checking.
#ifndef NO_CHECK_GLERROR
#define CHECK_GLERROR() checkGLError(__FILE__, __LINE__)
#else
#define CHECK_GLERROR() GL_NO_ERROR
#endif

inline int checkGLError(const char* f, const int line)
{
  GLuint err = glGetError();
  if (err != GL_NO_ERROR){
    std::cerr << f << ":" << line << ": OpenGL Error: '" << gluErrorString(err) <<"'"<<std::endl;
  }
  return err;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
/*inline void setSwapInterval(int interval)
{
#ifdef M_WINDOWS
#else
  Display *dpy = glXGetCurrentDisplay();
  GLXDrawable drawable = glXGetCurrentDrawable();
  unsigned int swap, maxSwap;
  const char* extensions = glXQueryExtensionsString(dpy,DefaultScreen(dpy));
  typedef void (*GLXSWAPINTERVALEXT)(Display*,GLXDrawable,int);
  GLXSWAPINTERVALEXT glXSwapIntervalEXT = NULL;

  if(strstr(extensions,"EXT_swap_control"))
  {
      glXSwapIntervalEXT = (GLXSWAPINTERVALEXT)glXGetProcAddress((const GLubyte*)"glXSwapIntervalEXT");
  }
  if (glXSwapIntervalEXT && drawable) {
      glXQueryDrawable(dpy, drawable, GLX_SWAP_INTERVAL_EXT, &swap);
      glXQueryDrawable(dpy, drawable, GLX_MAX_SWAP_INTERVAL_EXT,
                       &maxSwap);
      printf("Swap: %u (max: %u)\n",swap,maxSwap);
      if(interval>maxSwap)
        interval=maxSwap;
      if(interval==0)
        printf("Deactivate VSync...\n");
      if(interval==-1)
        printf("Enable Adaptive VSync...\n");
      glXSwapIntervalEXT(dpy, drawable, interval);
  }
#endif
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
inline float calculate_fps()
{
  static int frame_count = 0;
  static int current_time = 0, previous_time = 0;
  static float fps = 0;
  int time_interval = 0;
  //  Increase frame count
  frame_count++;

  //  Get the number of milliseconds since glutInit called
  //  (or first call to glutGet(GLUT ELAPSED TIME)).
  current_time = glutGet(GLUT_ELAPSED_TIME);

  //  Calculate time passed
  //int time_interval = current_time - previous_time;
  time_interval = current_time - previous_time;

  if (time_interval > 1000)
  {
    //  calculate the number of frames per second
    fps = frame_count / (time_interval / 1000.0f);

    //  Set time
    previous_time = current_time;

    //  Reset frame count
    frame_count = 0;
  }
  return fps;
}*/
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
inline void upload(GLuint buffer,
            char* h_data,
            const unsigned size,
            GLenum target,
            GLenum access)
{
  glBindBuffer(target, buffer);
  /*
   * We do not use glBufferData for data upload, since it would load into
   * host ram as pinned memory instead of loading into GPU RAM.
   * With glMapBuffer+memcpy we ensure to have our data located on GPU.
   */
  glBufferData(target, size, NULL, access);
  void* d_data = (void*) glMapBuffer(target, GL_READ_WRITE);
  if (d_data == NULL)
  {
    throw std::runtime_error("Could not map gpu buffer.");
  }
  memcpy(d_data, (const void*) h_data, size);
  if (!glUnmapBuffer(target))
  {
    throw std::runtime_error("Unmap buffer failed.");
  }
  d_data = NULL;
  // release buffer
  glBindBuffer(target, 0);
}

#endif /* GL_GLOBALS_H_ */

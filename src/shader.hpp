/*****************************************************************************/
/**
 * @brief OpenGL shader helper class.
 * @author Matthias Werner
 * @sa http://11235813tdd.blogspot.de/
 * @sa http://pages.cs.wisc.edu/~shenoy/
 *****************************************************************************/
#ifndef SHADER_H__
#define SHADER_H__

#include "gl_globals.hpp"
#include <string>

//Manages vertex and pixel shaders
class ShaderManager final
{
public:
  ShaderManager();
  ~ShaderManager();
  void load(const char * vertexshader,
            const char * pixelshader,
            const char * geoshader = NULL);
  void link();
  void bind();

  int getUniformVarID(const char * name);

  void setUniformVar(const char* name, int varValue);
  void setUniformVar(const char* name, int var[4]);
  void setUniformVar(const char* name, float varValue);
  void setUniformVar(const char* name, float var[4]);

  void setUniformMat4(const char* name, float* array);

  void unbind();
  void unload();
  bool isLoaded();

  GLuint programID() const
  {
    return programID_;
  }

private:
  GLuint loadShader(const char * filename, int type);
  void readEntireFile(std::string* content, const char * filename);

private:
  GLuint programID_;
  GLuint vertexShaderID_;
  GLuint pixelShaderID_;
  GLuint geoShaderID_;
  bool isLoaded_;

};

#endif

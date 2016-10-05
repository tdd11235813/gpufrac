/*****************************************************************************/
/**
 * @author Matthias Werner
 * @sa http://11235813tdd.blogspot.de/
 *****************************************************************************/
#include "shader.hpp"
#include <fstream>
#include <string>
using std::ifstream;

static char buffer[1024];
static int len = 0;

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
ShaderManager::ShaderManager()
    : programID_(0), vertexShaderID_(0), pixelShaderID_(0), geoShaderID_(0), isLoaded_(
        false)
{

}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
ShaderManager::~ShaderManager()
{
  if (programID_)
    glDeleteProgram(programID_);
}
//-----------------------------------------------------------------------------
//bundles a VS and a PS into a program
//-----------------------------------------------------------------------------
void ShaderManager::load(const char * vertexshader,
                         const char * pixelshader,
                         const char * geoshader)
{
  if (isLoaded_)
  {
    printf("Shader already loaded to application.\n");
    exit(1);
  }
  if (programID_ == 0)
    programID_ = glCreateProgram();
  if (vertexshader != NULL)
  {
    vertexShaderID_ = loadShader(vertexshader, GL_VERTEX_SHADER);
    glAttachShader(programID_, vertexShaderID_);
  }
  if (pixelshader != NULL)
  {
    pixelShaderID_ = loadShader(pixelshader, GL_FRAGMENT_SHADER);
    glAttachShader(programID_, pixelShaderID_);
  }
  if (geoshader != NULL)
  {
    geoShaderID_ = loadShader(geoshader, GL_GEOMETRY_SHADER);
    glAttachShader(programID_, geoShaderID_);

    glProgramParameteriEXT(programID_, GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS);
    glProgramParameteriEXT(programID_, GL_GEOMETRY_OUTPUT_TYPE_EXT,
                           GL_TRIANGLE_STRIP);
    glProgramParameteriEXT(programID_, GL_GEOMETRY_VERTICES_OUT_EXT, 4);
  }
  isLoaded_ = true;
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void ShaderManager::link()
{
  glLinkProgram(programID_);
  if(CHECK_GLERROR() != GL_NO_ERROR)
    throw std::runtime_error("Could not link program.");
  glGetProgramInfoLog(programID_, 1000, &len, buffer);
  if (len)
  {
    std::cerr << "Link error, program "<<programID_<<": "<< buffer << std::endl;
    throw std::runtime_error("Linking program failed.");
  }
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
GLuint ShaderManager::loadShader(const char *filename, int type)
{
  if (filename == NULL)
  {
    throw std::runtime_error("ERROR. No Filename given.");
  }
  GLhandleARB handle;

  // shader Compilation variable
  GLint result; // Compilation code result
  GLint errorLoglength;
  char* errorLogText;
  GLsizei actualErrorLogLength;

  handle = glCreateShaderObjectARB(type);
  if (!handle)
    throw std::runtime_error(std::string("Failed creating shader object from file: ")+filename);

  std::string content;
  readEntireFile(&content, filename);
  GLchar const *shader_source = content.c_str();
  GLint const shader_length = content.size();
  glShaderSource(handle, //The handle to our shader
      1, //The number of files.
      &shader_source, //An array of const char * data, which represents the source code of theshaders
      &shader_length);

  glCompileShaderARB(handle);

  //Compilation checking.
  glGetObjectParameterivARB(handle, GL_OBJECT_COMPILE_STATUS_ARB, &result);

  // If an error was detected.
  if (!result)
  {
    //Attempt to get the length of our error log.
    glGetObjectParameterivARB(handle, GL_OBJECT_INFO_LOG_LENGTH_ARB,
                              &errorLoglength);

    //Create a buffer to read compilation error message
    errorLogText = (char*) malloc(sizeof(char) * errorLoglength);

    //Used to get the final length of the log.
    glGetInfoLogARB(handle, errorLoglength, &actualErrorLogLength,
                    errorLogText);

    // Display errors.
    std::cerr<< errorLogText<<std::endl;

    // Free the buffer malloced earlier
    free(errorLogText);
    throw std::runtime_error(std::string("Shader compilation failed: ") + filename);
  }

  if (CHECK_GLERROR() != GL_NO_ERROR)
  {
    throw std::runtime_error("Could not create shader.");
  }

  return handle;
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
int ShaderManager::getUniformVarID(const char *name)
{
  glGetUniformLocation(programID_, name);
  if (CHECK_GLERROR() != GL_NO_ERROR)
  {
    fprintf(stderr, "Could not get uniform location '%s'.\n", name);
    return -1;
  }
  return glGetUniformLocation(programID_, name);
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void ShaderManager::setUniformVar(const char* name, int var[4])
{
  glUniform4i(getUniformVarID(name), var[0], var[1], var[2], var[3]);
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void ShaderManager::setUniformVar(const char* name, int var)
{
  glUniform1i(getUniformVarID(name), var);
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void ShaderManager::setUniformVar(const char* name, float var)
{
  glUniform1f(getUniformVarID(name), var);
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void ShaderManager::setUniformVar(const char* name, float var[4])
{
  glUniform4f(getUniformVarID(name), var[0], var[1], var[2], var[3]);
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void ShaderManager::setUniformMat4(const char* name, float* array)
{
  glUniformMatrix4fv(getUniformVarID(name), 1, GL_FALSE, array);
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void ShaderManager::bind()
{
  glUseProgram(programID_);
  glEnable(GL_VERTEX_PROGRAM_ARB);
  glEnable(GL_FRAGMENT_PROGRAM_ARB);
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void ShaderManager::unbind()
{
  glUseProgram(0);
  glDisable(GL_VERTEX_PROGRAM_ARB);
  glDisable(GL_FRAGMENT_PROGRAM_ARB);
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
bool ShaderManager::isLoaded()
{
  return isLoaded_;
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void ShaderManager::readEntireFile(std::string* content, const char * filename)
{
  // --- Read file
  std::string line;
  ifstream myfile(filename);
  if (myfile.is_open())
  {
    while (myfile.good())
    {
      getline(myfile, line);
      *content += line + "\n";
    }
    myfile.close();
  }
  else
  {
    printf("Could not open file '%s'.\n", filename);
    exit(1);
  }
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void ShaderManager::unload()
{
  if (programID_ == 0)
  {
    printf("Error occurred on unloading Shader.\n");
    exit(1);
  }
  if (vertexShaderID_)
    glDetachShader(programID_, vertexShaderID_);
  if (pixelShaderID_)
    glDetachShader(programID_, pixelShaderID_);
  if (geoShaderID_)
    glDetachShader(programID_, geoShaderID_);
  isLoaded_ = false;
}

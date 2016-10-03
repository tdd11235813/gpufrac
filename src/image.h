#ifndef __IMAGE_H__
#define __IMAGE_H__

/*
 Original source from http://danielbeard.wordpress.com/2011/06/06/image-saving-code-c/
 with Bugfixes and some minor changes (2013/05/09).
 */

//includes
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

//data structures
struct Colour
{
  unsigned char r, g, b, a;
};

class TGAImage
{

public:

  //Constructor
  TGAImage() = default;
  TGAImage(const std::string& path) : path_(path) {}
  ~TGAImage();

  //Overridden Constructor
  TGAImage(size_t width, size_t height);
  TGAImage(const std::string& path, size_t width, size_t height);

  //Set all pixels at once
  void setAllPixels(Colour *pixels);

  //set individual pixels
  void setPixel(Colour inputcolor, size_t xposition, size_t yposition);

  bool exists(std::string filename);
  void save(std::string filename);
  std::string saveByPrefix(std::string prefix);
  static std::string saveOpenGLBuffer(size_t width,
                                      size_t height,
                                      std::string outputDir="./",
                                      std::string prefix="img_");
  static std::string savePBO(int buffer_id,
                             size_t width,
                             size_t height,
                             std::string outputDir="./",
                             std::string prefix="img_");

//General getters and setters

  void setWidth(size_t width);
  void setHeight(size_t height);

  size_t width() const
  {
    return width_;
  }
  size_t height() const
  {
    return height_;
  }
  std::string path() const
  {
    return path_;
  }

  unsigned char* rawPointer()
  {
    return reinterpret_cast<unsigned char*>(pixels_);
  }

private:

  //store the pixels
  Colour *pixels_ = nullptr;
  std::string path_ = "./";
  size_t height_ = 0;
  size_t width_ = 0;

  //convert 2D to 1D indexing
  size_t convert2dto1d(size_t x, size_t y) {
    return width_ * y + x;
  }
};

#endif

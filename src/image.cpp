#include "image.h"
#include "gl_globals.h"

TGAImage::~TGAImage()
{
  if (pixels_)
    delete[] pixels_;
}

//Overwritten Constructor
TGAImage::TGAImage(const std::string& path, size_t width, size_t height)
  : path_(path), width_(width), height_(height)
{
  pixels_ = new Colour[width_ * height_];
}

TGAImage::TGAImage(size_t width, size_t height)
 : TGAImage("./", width, height)
{}

void TGAImage::setWidth(size_t width)
{
  width_ = width;
  if (pixels_)
    delete[] pixels_;
  pixels_ = new Colour[width_ * height_];
}
void TGAImage::setHeight(size_t height)
{
  height_ = height;
  if (pixels_)
    delete[] pixels_;
  pixels_ = new Colour[width_ * height_];
}
//Set all pixels at once
void TGAImage::setAllPixels(Colour *pixels)
{
  pixels_ = pixels;
}

//Set indivdual pixels
void TGAImage::setPixel(Colour inputcolor, size_t x, size_t y)
{
  pixels_[convert2dto1d(x, y)] = inputcolor;
}

std::string TGAImage::saveOpenGLBuffer(size_t width, size_t height, string outputDir, string prefix)
{
  static int counter = 0;
  std::string floc = "";
  outputDir += "/";
  TGAImage img(outputDir, width, height);
  unsigned char* rawptr = img.rawPointer();
  if(!rawptr)
  {
    throw runtime_error("Could not get image raw pointer.");
  }
  glReadBuffer(GL_BACK);
  glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)(rawptr));
  if (CHECK_GLERROR() == GL_NO_ERROR && rawptr)
  {
    char imgfile[128];
    do{
      ++counter;
      sprintf(imgfile, "%s%04i", prefix.c_str(), counter);
    }while(img.exists(imgfile));
    try{
      img.save(imgfile);
      floc = img.path() + imgfile + ".tga";
    }catch(const std::runtime_error& e){
      --counter;
      throw std::runtime_error(std::string(e.what())+"\nDirectory could not be found.");
    }
  }else
    throw runtime_error("Could not save image.");
  return floc;
}

bool TGAImage::exists(string filename) {
  std::string floc = path_ + filename + ".tga";
  ifstream f(floc);
  return f.good();
}

void TGAImage::save(string filename)
{
  //Error checking
  if (width_ == 0 || height_ == 0)
  {
    throw std::runtime_error("Image size is not set properly");
  }

  std::string floc = path_ + filename + ".tga";
  ofstream o(floc, ios::out | ios::binary);
  if (o.good() == false)
  {
    throw std::runtime_error("TGAImage::WriteImage: Could not open location: " + floc);
  }

  //Write the header
  o.put(0);
  o.put(0);
  o.put(2); /* uncompressed RGB */
  o.put(0);
  o.put(0);
  o.put(0);
  o.put(0);
  o.put(0);
  o.put(0); /* X origin */
  o.put(0);
  o.put(0); /* y origin */
  o.put(0);
  o.put((width_ & 0x00FF));
  o.put((width_ & 0xFF00)>>8);
  o.put((height_ & 0x00FF));
  o.put((height_ & 0xFF00)>>8);
  o.put(24);
  o.put(0);

  //Write the pixel data
  for (int i = 0; i < height_ * width_; i++)
  {
    o.put(pixels_[i].b);
    o.put(pixels_[i].g);
    o.put(pixels_[i].r);
  }

  //close the file
  o.close();
}

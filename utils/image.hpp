#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <string>
#define NUM_CHANNELS 3 // We will only support RGB images

class Image
{
  public:
    Image();
    ~Image();

    bool load( const std::string & filename );
    bool save( const std::string & filename ) const;

    // Used to create an image from the buffer returned by loadRawImage
    bool createFromRawImage( const unsigned char * rawBuffer, int rawByteSize );

    // Used to create an empty image
    void createEmpty( const int width, const int height );

    unsigned char * data() const { return _data; }

    int width() const { return _width; }
    int height() const { return _height; }
    int channels() const { return NUM_CHANNELS; } // Number of colors, in this case 3 (RGB)
    int byteSize() const { return _width * _height * NUM_CHANNELS; }

    Image copy() const; // Deep copy the image data to a new image

  private:
    unsigned char * _data;
    int             _width;
    int             _height;
};
#endif // IMAGE_HPP
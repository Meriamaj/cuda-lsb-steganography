#include "image.hpp"

#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

Image::Image() : _data( nullptr ), _width( 0 ), _height( 0 ) {}

Image::~Image() { delete[] _data; }

bool Image::createFromRawImage( const unsigned char * rawBuffer, const int32_t rawByteSize )
{
    int channels;

    _data = stbi_load_from_memory( rawBuffer, rawByteSize, &_width, &_height, &channels, NUM_CHANNELS );

    return _data != nullptr;
}

bool Image::load( const std::string & filename )
{
    int channels;

    _data = stbi_load( filename.c_str(), &_width, &_height, &channels, NUM_CHANNELS );

    return _data != nullptr;
}

void Image::createEmpty( const int32_t width, const int32_t height )
{
    _data   = new unsigned char[ width * height * NUM_CHANNELS ];
    _width  = width;
    _height = height;
}

bool Image::save( const std::string & filename ) const { return stbi_write_png( filename.c_str(), _width, _height, NUM_CHANNELS, _data, _width * NUM_CHANNELS ); }

Image Image::copy() const
{
    Image copy;
    copy.createEmpty( _width, _height );
    memcpy( copy.data(), _data, byteSize() );
    return copy;
}
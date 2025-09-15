#include <fstream>

#include "raw_image_loader.hpp"

// Loads the raw image (not the pixel data)
unsigned char * loadRawImage( const std::string & path, int & byteSize )
{
    std::ifstream file;
    file.open( path );

    if ( !file.is_open() )
        return nullptr;

    file.seekg( 0, std::ios::end );
    byteSize             = static_cast<int>( file.tellg() );
    unsigned char * data = new unsigned char[ byteSize ];

    file.seekg( 0, std::ios::beg );
    file.read( reinterpret_cast<char *>( data ), byteSize );

    file.close();

    return data;
}
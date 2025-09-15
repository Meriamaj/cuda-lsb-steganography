#ifndef STEGANOGRAPHY_HPP
#define STEGANOGRAPHY_HPP
#include "image.hpp"
#include <string>

bool naive_encode(Image& carrier, const std::string& secret_image_path, size_t& expected_end_pos);
bool naive_find_sentinels(const Image& original, const Image& modified, size_t& start_pos, size_t& end_pos);
bool naive_decrypt(const Image& modified, size_t start_pos, size_t end_pos, Image& secret);

#endif

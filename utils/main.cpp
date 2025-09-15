#include "image.hpp"
#include "raw_image_loader.hpp"
#include "steganography.hpp"
#include <iostream>

int main() {
   // Étape d'encodage
    Image carrier;
    if (!carrier.load("../images/boule.png")) return 1;
    
    size_t end_pos;
    if (!naive_encode(carrier, "../images/malice.png", end_pos)) return 1;
    if (!carrier.save("../images/stego/boule_malice_stego.png")) return 1;

    // Étape d'extraction
    Image original, modified;
    if (!original.load("../images/agile.png")) return 1;
    if (!modified.load("../images/stego/agile_stego.png")) return 1;

    size_t start_pos, end_pos_extract;
    if (!naive_find_sentinels(original, modified, start_pos, end_pos_extract)) return 1;

    Image secret;
    if (!naive_decrypt(modified, start_pos, end_pos_extract, secret)) return 1;
    if (!secret.save("../images/extracted/secreteeeee.png")) return 1;

    std::cout << "Operation completed successfully!" << std::endl;
    return 0;

}

#include "image.hpp"
#include "raw_image_loader.hpp"
#include "chronoCPU.hpp"
#include <cassert>
#include <iostream>
#include <vector>

bool naive_encode(Image& carrier, const std::string& secret_image_path, size_t& expected_end_pos) {
    ChronoCPU chrono;
    chrono.start();

    bool result = false;

    int secret_byte_size;
    unsigned char* secret_data = loadRawImage(secret_image_path, secret_byte_size);
    if (!secret_data) {
        std::cerr << "Erreur : impossible de charger l'image secrète." << std::endl;
    } else {
        size_t carrier_capacity = carrier.byteSize();
        size_t required_bits = secret_byte_size * 8 + 2;
        std::cout << "Taille image secrète (octets) : " << secret_byte_size << std::endl;
        std::cout << "Bits requis (données + sentinelles) : " << required_bits << std::endl;
        std::cout << "Capacité porteuse (octets) : " << carrier_capacity << std::endl;

        if (required_bits > carrier_capacity) {
            std::cerr << "Erreur : l'image porteuse est trop petite." << std::endl;
            delete[] secret_data;
        } else {
            unsigned char* carrier_data = carrier.data();
            carrier_data[0] = (carrier_data[0] & 0xFE) | (~(carrier_data[0] & 0x01) & 0x01);
            std::cout << "Sentinelle de début placée à l'index 0, valeur : " << (int)carrier_data[0] << std::endl;

            size_t carrier_index = 1;
            for (int i = 0; i < secret_byte_size; ++i) {
                unsigned char byte = secret_data[i];
                for (int bit = 7; bit >= 0; --bit) {
                    unsigned char secret_bit = (byte >> bit) & 0x01;
                    carrier_data[carrier_index] = (carrier_data[carrier_index] & 0xFE) | (secret_bit & 0x01);
                    ++carrier_index;
                }
            }

            if (carrier_index < carrier_capacity) {
                carrier_data[carrier_index] = (carrier_data[carrier_index] & 0xFE) | (~(carrier_data[carrier_index] & 0x01) & 0x01);
                std::cout << "Sentinelle de fin placée à l'index " << carrier_index << ", valeur : " << (int)carrier_data[carrier_index] << std::endl;
                expected_end_pos = carrier_index;
                result = true;
            } else {
                std::cerr << "Erreur : pas assez d'espace pour la sentinelle de fin." << std::endl;
            }
            std::cout << "Octet à l'index 3 : " << (int)carrier_data[3] << std::endl;
            delete[] secret_data;
        }
    }

    chrono.stop();
    std::cout << "Temps d'exécution de naive_encode : " << chrono.elapsedTime() << " millisecondes" << std::endl;

    return result;
}

bool naive_find_sentinels(const Image& original, const Image& modified, size_t& start_pos, size_t& end_pos) {
    ChronoCPU chrono;
    chrono.start();

    bool result = false;

    if (original.byteSize() != modified.byteSize()) {
        std::cerr << "Erreur : les images n'ont pas la même taille." << std::endl;
    } else {
        const unsigned char* original_data = original.data();
        const unsigned char* modified_data = modified.data();
        size_t byte_size = original.byteSize();

        start_pos = end_pos = 0;
        bool found_start = false;
        std::vector<size_t> differences;

        // Trouver les différences de LSB entre original et modifié
        for (size_t i = 0; i < byte_size; ++i) {
            unsigned char original_lsb = original_data[i] & 0x01;
            unsigned char modified_lsb = modified_data[i] & 0x01;

            if (original_lsb != modified_lsb) {
                differences.push_back(i);
                if (!found_start) {
                    start_pos = i;
                    found_start = true;
                    std::cout << "Sentinelle de début trouvée à l'index " << start_pos << std::endl;
                }
            }
        }

        if (!found_start) {
            std::cerr << "Erreur : aucune sentinelle de début trouvée." << std::endl;
        } else {
            for (size_t i = 1; i < differences.size(); ++i) {
                size_t candidate_end_pos = differences[i];
                size_t data_bits = candidate_end_pos - start_pos - 1;

                // Vérifier si les données sont alignées sur un octet et assez grandes
                if (data_bits % 8 == 0 && data_bits >= 64) {
                    size_t secret_byte_size = data_bits / 8;
                    unsigned char* secret_data = new unsigned char[secret_byte_size];

                    // Extraire les bits de données entre les sentinelles
                    for (size_t j = 0; j < secret_byte_size; ++j) {
                        unsigned char byte = 0;
                        for (int bit = 7; bit >= 0; --bit) {
                            size_t pos = start_pos + 1 + j * 8 + (7 - bit);
                            unsigned char lsb = modified_data[pos] & 0x01;
                            byte |= (lsb << bit);
                        }
                        secret_data[j] = byte;
                    }

                    // Créer une image à partir des données extraites
                    Image secret;
                    if (secret.createFromRawImage(secret_data, secret_byte_size)) {
                        end_pos = candidate_end_pos;
                        std::cout << "Succès : image secrète valide extraite avec sentinelle de fin à l'index " << end_pos << std::endl;
                        result = true;
                        delete[] secret_data;
                        break;
                    }
                    delete[] secret_data;
                }
            }
            if (!result) {
                std::cerr << "Erreur : aucune sentinelle de fin valide trouvée après " << differences.size() << " différences." << std::endl;
            }
        }
    }

    chrono.stop();
    std::cout << "Temps d'exécution de naive_find_sentinels : " << chrono.elapsedTime() << " millisecondes" << std::endl;

    return result;
}

bool naive_decrypt(const Image& modified, size_t start_pos, size_t end_pos, Image& secret) {
    ChronoCPU chrono;
    chrono.start();

    bool result = false;

    if (start_pos >= end_pos || end_pos > modified.byteSize()) {
        std::cerr << "Erreur : positions des sentinelles invalides." << std::endl;
    } else if ((end_pos - start_pos - 1) % 8 != 0) {
        std::cerr << "Erreur : taille des données non alignée sur un octet." << std::endl;
    } else {
        size_t data_bits = end_pos - start_pos - 1;
        size_t secret_byte_size = data_bits / 8;

        unsigned char* secret_data = new unsigned char[secret_byte_size];
        const unsigned char* modified_data = modified.data();

        for (size_t i = 0; i < secret_byte_size; ++i) {
            unsigned char byte = 0;
            for (int bit = 7; bit >= 0; --bit) {
                size_t pos = start_pos + 1 + i * 8 + (7 - bit);
                unsigned char lsb = modified_data[pos] & 0x01;
                byte |= (lsb << bit);
            }
            secret_data[i] = byte;
        }

        result = secret.createFromRawImage(secret_data, secret_byte_size);
        delete[] secret_data;
    }

    chrono.stop();
    std::cout << "Temps d'exécution de naive_decrypt : " << chrono.elapsedTime() << " millisecondes" << std::endl;

    return result;
}

#include "image.hpp"
#include "raw_image_loader.hpp"
#include "chronoGPU.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "commonCUDA.hpp"

// Déclarations anticipées des fonctions
bool cuda_naive_encode(Image& carrier, const std::string& secret_image_path, size_t& expected_end_pos);
bool cuda_naive_find_sentinels(const Image& original, const Image& modified, size_t& start_pos, size_t& end_pos);
bool cuda_naive_decrypt(const Image& modified, size_t start_pos, size_t end_pos, Image& secret, ChronoGPU& chrono);

// Kernel CUDA pour l'encodage
__global__ void cuda_encode_kernel(unsigned char* carrier_data, unsigned char* secret_data, int secret_byte_size, size_t carrier_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= carrier_size)
        return;

    // Sentinelle de début (premier octet)
    if (idx == 0) {
        carrier_data[0] = (carrier_data[0] & 0xFE) | (~(carrier_data[0] & 0x01) & 0x01);
    }

    // Encodage des bits de l'image secrète
    if (idx >= 1 && idx < secret_byte_size * 8 + 1) {
        int secret_byte_idx = (idx - 1) / 8;
        int bit_idx = 7 - ((idx - 1) % 8);
        unsigned char secret_bit = (secret_data[secret_byte_idx] >> bit_idx) & 0x01;
        carrier_data[idx] = (carrier_data[idx] & 0xFE) | (secret_bit & 0x01);
    }

    // Sentinelle de fin
    if (idx == secret_byte_size * 8 + 1 && idx < carrier_size) {
        carrier_data[idx] = (carrier_data[idx] & 0xFE) | (~(carrier_data[idx] & 0x01) & 0x01);
    }
}

// Kernel CUDA pour trouver les différences de LSB
__global__ void cuda_find_differences_kernel(const unsigned char* original_data, const unsigned char* modified_data, 
                                            size_t byte_size, int* differences, int* diff_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < byte_size) {
        unsigned char original_lsb = original_data[idx] & 0x01;
        unsigned char modified_lsb = modified_data[idx] & 0x01;
        if (original_lsb != modified_lsb) {
            int pos = atomicAdd(diff_count, 1);
            if (pos < byte_size) {  // Protection contre le dépassement
                differences[pos] = idx;
            }
        }
    }
}

// Kernel CUDA pour l'extraction
__global__ void cuda_decrypt_kernel(const unsigned char* modified_data, size_t start_pos, size_t secret_byte_size, unsigned char* secret_data) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < secret_byte_size) {
        unsigned char byte = 0;
        for (int bit = 0; bit < 8; ++bit) {
            size_t pos = start_pos + 1 + idx * 8 + bit;
            unsigned char lsb = modified_data[pos] & 0x01;
            byte |= (lsb << (7 - bit));
        }
        secret_data[idx] = byte;
    }
}

// Version CUDA de naive_encode
bool cuda_naive_encode(Image& carrier, const std::string& secret_image_path, size_t& expected_end_pos) {
    int secret_byte_size;
    unsigned char* secret_data = loadRawImage(secret_image_path, secret_byte_size);
    if (!secret_data) {
        std::cerr << "Erreur : impossible de charger l'image secrète." << std::endl;
        return false;
    }

    size_t carrier_size = carrier.byteSize();
    size_t required_bits = secret_byte_size * 8 + 2;
    std::cout << "Taille image secrète (octets) : " << secret_byte_size << std::endl;
    std::cout << "Bits requis (données + sentinelles) : " << required_bits << std::endl;
    std::cout << "Capacité porteuse (octets) : " << carrier_size << std::endl;

    if (required_bits > carrier_size) {
        std::cerr << "Erreur : l'image porteuse est trop petite." << std::endl;
        delete[] secret_data;
        return false;
    }

    // Allouer la mémoire GPU
    unsigned char *d_carrier_data, *d_secret_data;
    HANDLE_ERROR(cudaMalloc(&d_carrier_data, carrier_size * sizeof(unsigned char)));
    HANDLE_ERROR(cudaMalloc(&d_secret_data, secret_byte_size * sizeof(unsigned char)));

    // Copier les données vers le GPU
    HANDLE_ERROR(cudaMemcpy(d_carrier_data, carrier.data(), carrier_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_secret_data, secret_data, secret_byte_size * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Configurer la grille et les blocs
    int threads_per_block = 256;
    int blocks = (carrier_size + threads_per_block - 1) / threads_per_block;

    // Utiliser ChronoGPU pour mesurer le temps
    ChronoGPU chrono;
    chrono.start();
    
    // Lancer le kernel
    cuda_encode_kernel<<<blocks, threads_per_block>>>(d_carrier_data, d_secret_data, secret_byte_size, carrier_size);
    HANDLE_ERROR(cudaGetLastError());
    
    // Arrêter le chronomètre
    chrono.stop();
    std::cout << "Temps d'exécution du kernel d'encodage: " << chrono.elapsedTime() << " ms" << std::endl;

    HANDLE_ERROR(cudaDeviceSynchronize());

    // Récupérer les données modifiées
    HANDLE_ERROR(cudaMemcpy(carrier.data(), d_carrier_data, carrier_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Débogage
    std::cout << "Sentinelle de début placée à l'index 0, LSB = " << (int)(carrier.data()[0] & 0x01) << std::endl;
    std::cout << "Sentinelle de fin placée à l'index " << secret_byte_size * 8 + 1 
              << ", LSB = " << (int)(carrier.data()[secret_byte_size * 8 + 1] & 0x01) << std::endl;

    // Libérer la mémoire
    HANDLE_ERROR(cudaFree(d_carrier_data));
    HANDLE_ERROR(cudaFree(d_secret_data));
    delete[] secret_data;

    expected_end_pos = secret_byte_size * 8 + 1;
    return true;
}

// Version CUDA de naive_find_sentinels
bool cuda_naive_find_sentinels(const Image& original, const Image& modified, size_t& start_pos, size_t& end_pos) {
    if (original.byteSize() != modified.byteSize()) {
        std::cerr << "Erreur : les images n'ont pas la même taille." << std::endl;
        return false;
    }

    size_t byte_size = original.byteSize();
    int* d_differences;
    int* d_diff_count;
    int h_diff_count = 0;

    // Allouer la mémoire GPU
    HANDLE_ERROR(cudaMalloc(&d_differences, byte_size * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_diff_count, sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_diff_count, &h_diff_count, sizeof(int), cudaMemcpyHostToDevice));

    // Allouer et copier les données des images
    unsigned char *d_original_data, *d_modified_data;
    HANDLE_ERROR(cudaMalloc(&d_original_data, byte_size * sizeof(unsigned char)));
    HANDLE_ERROR(cudaMalloc(&d_modified_data, byte_size * sizeof(unsigned char)));
    HANDLE_ERROR(cudaMemcpy(d_original_data, original.data(), byte_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_modified_data, modified.data(), byte_size * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Configurer la grille et les blocs
    int threads_per_block = 256;
    int blocks = (byte_size + threads_per_block - 1) / threads_per_block;

    // Utiliser ChronoGPU pour mesurer le temps
    ChronoGPU chrono;
    chrono.start();

    // Lancer le kernel
    cuda_find_differences_kernel<<<blocks, threads_per_block>>>(d_original_data, d_modified_data, 
                                                                byte_size, d_differences, d_diff_count);
    HANDLE_ERROR(cudaGetLastError());

    // Arrêter le chronomètre
    chrono.stop();
    std::cout << "Temps d'exécution du kernel de recherche des sentinelles: " << chrono.elapsedTime() << " ms" << std::endl;

    HANDLE_ERROR(cudaDeviceSynchronize());

    // Récupérer le nombre de différences
    HANDLE_ERROR(cudaMemcpy(&h_diff_count, d_diff_count, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_diff_count < 2) {
        std::cerr << "Erreur : moins de 2 différences trouvées, pas de sentinelles." << std::endl;
        HANDLE_ERROR(cudaFree(d_differences));
        HANDLE_ERROR(cudaFree(d_diff_count));
        HANDLE_ERROR(cudaFree(d_original_data));
        HANDLE_ERROR(cudaFree(d_modified_data));
        return false;
    }

    // Récupérer les différences
    std::vector<int> differences(h_diff_count);
    HANDLE_ERROR(cudaMemcpy(differences.data(), d_differences, h_diff_count * sizeof(int), cudaMemcpyDeviceToHost));

    // Libérer la mémoire GPU
    HANDLE_ERROR(cudaFree(d_differences));
    HANDLE_ERROR(cudaFree(d_diff_count));
    HANDLE_ERROR(cudaFree(d_original_data));
    HANDLE_ERROR(cudaFree(d_modified_data));

    // Trier les différences
    std::sort(differences.begin(), differences.end());

    // Vérifier la sentinelle de début
    if (differences[0] != 0) {
        std::cerr << "Erreur : la sentinelle de début n'est pas à l'index 0 (trouvée à " 
                  << differences[0] << ")." << std::endl;
        return false;
    }

    start_pos = differences[0];
    std::cout << "Sentinelle de début confirmée à l'index " << start_pos << std::endl;

    // Créer une instance de ChronoGPU pour le décodage
    ChronoGPU decrypt_chrono;
    for (size_t i = 1; i < differences.size(); ++i) {
        size_t candidate_end_pos = differences[i];
        size_t data_bits = candidate_end_pos - start_pos - 1;
        
        if (data_bits % 8 == 0 && data_bits >= 64) {
            end_pos = candidate_end_pos;
            if (i > 1 && (end_pos - differences[i-1] > 100)) {
                std::cout << "Attention: grand écart entre les différences, possible faux positif." << std::endl;
            }

            Image secret;
            if (cuda_naive_decrypt(modified, start_pos, end_pos, secret, decrypt_chrono)) {
         
                return true;
            } 
        }
    }

    std::cerr << "Erreur : aucune sentinelle de fin valide trouvée." << std::endl;
    return false;
}

// Version CUDA de naive_decrypt
bool cuda_naive_decrypt(const Image& modified, size_t start_pos, size_t end_pos, Image& secret, ChronoGPU& chrono) {
    if (start_pos >= end_pos || end_pos > modified.byteSize()) {
        std::cerr << "Erreur : positions des sentinelles invalides." << std::endl;
        return false;
    }

    size_t data_bits = end_pos - start_pos - 1;
    if (data_bits % 8 != 0) {
        std::cerr << "Erreur : taille des données non alignée sur un octet." << std::endl;
        return false;
    }
    size_t secret_byte_size = data_bits / 8;

    // Allouer la mémoire GPU
    unsigned char *d_modified_data, *d_secret_data;
    HANDLE_ERROR(cudaMalloc(&d_modified_data, modified.byteSize() * sizeof(unsigned char)));
    HANDLE_ERROR(cudaMalloc(&d_secret_data, secret_byte_size * sizeof(unsigned char)));

    // Copier les données vers le GPU
    HANDLE_ERROR(cudaMemcpy(d_modified_data, modified.data(), modified.byteSize() * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Configurer la grille et les blocs
    int threads_per_block = 256;
    int blocks = (secret_byte_size + threads_per_block - 1) / threads_per_block;

    // Utiliser ChronoGPU pour mesurer le temps
    chrono.start();

    // Lancer le kernel
    cuda_decrypt_kernel<<<blocks, threads_per_block>>>(d_modified_data, start_pos, secret_byte_size, d_secret_data);
    HANDLE_ERROR(cudaGetLastError());

    // Arrêter le chronomètre
    chrono.stop();

    HANDLE_ERROR(cudaDeviceSynchronize());

    // Récupérer les données secrètes
    unsigned char* secret_data = new unsigned char[secret_byte_size];
    HANDLE_ERROR(cudaMemcpy(secret_data, d_secret_data, secret_byte_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Libérer la mémoire GPU
    HANDLE_ERROR(cudaFree(d_modified_data));
    HANDLE_ERROR(cudaFree(d_secret_data));

    // Vérifier la signature de l'image
    bool signature_ok = true;
    if (secret_byte_size > 8) {
        const unsigned char PNG_SIG[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
        const unsigned char JPEG_SIG[] = {0xFF, 0xD8};
        
        bool is_png = true;
        for (int i = 0; i < 8; i++) {
            if (secret_data[i] != PNG_SIG[i]) {
                is_png = false;
                break;
            }
        }
        
        bool is_jpeg = secret_data[0] == JPEG_SIG[0] && secret_data[1] == JPEG_SIG[1];
        signature_ok = is_png || is_jpeg;
        
        if (!signature_ok) {
            std::cout << "Attention: signature d'image non reconnue, les données pourraient être corrompues." << std::endl;
        }
    }

    // Reconstruire l'image
    bool success = secret.createFromRawImage(secret_data, secret_byte_size);
    delete[] secret_data;

    if (!success) {
        return false;
    }

    return true;
}

int main() {
    // Étape 1 : Encodage
    Image carrier;
    if (!carrier.load("../images/boule.png")) {
        std::cerr << "Erreur : impossible de charger l'image porteuse." << std::endl;
        return 1;
    }
    std::cout << "Dimensions porteuse : " << carrier.width() << "x" << carrier.height() << ", capacité : " << carrier.byteSize() << " octets" << std::endl;

    size_t end_pos;
    if (!cuda_naive_encode(carrier, "../images/malice.png", end_pos)) {
        std::cerr << "Erreur : impossible d'encoder l'image secrète." << std::endl;
        return 1;
    }
    
    std::string output_path = "../images/stego/boule_malice_stego.png";
    if (!carrier.save(output_path)) {
        std::cerr << "Erreur : impossible de sauvegarder l'image stéganographiée." << std::endl;
        return 1;
    }
    std::cout << "Succès : l'image stéganographiée a été créée sous " << output_path << std::endl;
    std::cout << "Position attendue de la sentinelle de fin: " << end_pos << std::endl;

    // Étape 2 : Extraction
    Image original;
    if (!original.load("../images/boule.png")) {
        std::cerr << "Erreur : impossible de charger l'image porteuse originale." << std::endl;
        return 1;
    }
    std::cout << "Dimensions porteuse originale : " << original.width() << "x" << original.height() << ", capacité : " << original.byteSize() << " octets" << std::endl;

    Image modified;
    if (!modified.load(output_path)) {
        std::cerr << "Erreur : impossible de charger l'image stéganographiée." << std::endl;
        return 1;
    }
    std::cout << "Dimensions image stéganographiée : " << modified.width() << "x" << modified.height() << ", capacité : " << modified.byteSize() << " octets" << std::endl;

    size_t start_pos, end_pos_extract;
    if (!cuda_naive_find_sentinels(original, modified, start_pos, end_pos_extract)) {
        std::cerr << "Erreur lors de la recherche des sentinelles." << std::endl;
        return 1;
    }

    // Vérifier que les positions de fin correspondent
    if (end_pos != end_pos_extract) {
        std::cout << "Attention: position de fin extraite (" << end_pos_extract 
                  << ") différente de la position attendue (" << end_pos << ")" << std::endl;
    }

    Image secret;
    ChronoGPU decrypt_chrono; // Créer une instance pour le main
    if (!cuda_naive_decrypt(modified, start_pos, end_pos_extract, secret, decrypt_chrono)) {
        std::cerr << "Erreur lors du décodage." << std::endl;
        return 1;
    }
    std::cout << "Temps d'exécution du kernel de décodage: " << decrypt_chrono.elapsedTime() << " ms" << std::endl;

    std::string extracted_path = "../images/extracted/image_secrete_extraiteeeeee.png";
    if (!secret.save(extracted_path)) {
        std::cerr << "Erreur lors de la sauvegarde de l'image extraite." << std::endl;
        return 1;
    }
    std::cout << "Succès : l'image secrète a été extraite sous " << extracted_path << std::endl;

    return 0;
}
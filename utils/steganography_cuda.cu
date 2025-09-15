#include "image.hpp"
#include "raw_image_loader.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Erreur CUDA: " << cudaGetErrorString(err) << " à la ligne " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Déclaration anticipée
bool cuda_naive_decrypt(const Image& modified, size_t start_pos, size_t end_pos, Image& secret);

// Noyau CUDA pour trouver les différences de LSB
__global__ void cuda_find_differences_kernel(const unsigned char* original, const unsigned char* modified, size_t size, int* differences, int* diff_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if ((original[idx] & 0x01) != (modified[idx] & 0x01)) {
            int pos = atomicAdd(diff_count, 1);
            differences[pos] = idx;
        }
    }
    __syncthreads(); // Synchronisation pour garantir que toutes les écritures sont terminées
}

// Noyau CUDA pour décrypter l'image secrète
__global__ void cuda_decrypt_kernel(const unsigned char* modified, size_t start_pos, size_t secret_bit_size, unsigned char* secret_data) {
    extern __shared__ unsigned char shared_bits[]; // Mémoire partagée pour les bits par bloc
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Chaque thread écrit son bit dans la mémoire partagée
    if (idx < secret_bit_size) {
        size_t pos = start_pos + 1 + idx;
        shared_bits[tid] = modified[pos] & 0x01;
    } else {
        shared_bits[tid] = 0;
    }
    __syncthreads(); // Synchronisation pour garantir que tous les threads ont écrit

    // Les threads responsables des bits d'un octet combinent les bits
    if (idx < secret_bit_size) {
        size_t byte_idx = idx / 8;
        int bit_idx = 7 - (idx % 8); // MSB à LSB
        if (shared_bits[tid]) {
            secret_data[byte_idx] |= (1 << bit_idx);
        }
    }
    __syncthreads(); // Synchronisation pour garantir que toutes les écritures sont terminées
}

bool cuda_naive_find_sentinels(const Image& original, const Image& modified, size_t& start_pos, size_t& end_pos) {
    if (original.byteSize() != modified.byteSize()) {
        std::cerr << "Erreur : les images n'ont pas la même taille." << std::endl;
        std::cerr << "Taille originale : " << original.byteSize() << " octets, Taille modifiée : " << modified.byteSize() << " octets" << std::endl;
        return false;
    }

    size_t size = original.byteSize();
    std::cout << "Taille de l'image : " << size << " octets" << std::endl;
    std::vector<int> differences(size); // Taille max pour éviter débordements
    int h_diff_count = 0;

    // Allocation mémoire GPU
    unsigned char *d_original, *d_modified;
    int *d_differences, *d_diff_count;
    CUDA_CHECK(cudaMalloc(&d_original, size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_modified, size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_differences, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_diff_count, sizeof(int)));

    // Initialiser diff_count à 0
    CUDA_CHECK(cudaMemset(d_diff_count, 0, sizeof(int)));

    // Copier les données
    CUDA_CHECK(cudaMemcpy(d_original, original.data(), size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_modified, modified.data(), size * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Lancer le noyau
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "Lancement du noyau avec " << blocks << " blocs et " << threadsPerBlock << " threads par bloc" << std::endl;
    cuda_find_differences_kernel<<<blocks, threadsPerBlock>>>(d_original, d_modified, size, d_differences, d_diff_count);
    CUDA_CHECK(cudaDeviceSynchronize()); // Attendre que le noyau soit terminé

    // Copier le nombre de différences
    CUDA_CHECK(cudaMemcpy(&h_diff_count, d_diff_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Nombre de différences détectées : " << h_diff_count << std::endl;
    if (h_diff_count < 2) {
        std::cerr << "Erreur : pas assez de différences trouvées (" << h_diff_count << ")." << std::endl;
        CUDA_CHECK(cudaFree(d_original));
        CUDA_CHECK(cudaFree(d_modified));
        CUDA_CHECK(cudaFree(d_differences));
        CUDA_CHECK(cudaFree(d_diff_count));
        return false;
    }

    // Copier les différences
    differences.resize(h_diff_count);
    CUDA_CHECK(cudaMemcpy(differences.data(), d_differences, h_diff_count * sizeof(int), cudaMemcpyDeviceToHost));

    // Trier les différences
    std::sort(differences.begin(), differences.end());

    // Débogage : afficher toutes les différences
    std::cout << "Différences trouvées (" << h_diff_count << ") : ";
    for (int i = 0; i < h_diff_count; ++i) {
        std::cout << differences[i] << " ";
    }
    std::cout << std::endl;

    start_pos = differences[0];
    std::cout << "Sentinelle de début trouvée à l'index " << start_pos << std::endl;

    // Tester les sentinelles de fin
    for (size_t i = 1; i < differences.size(); ++i) {
        size_t candidate_end_pos = differences[i];
        size_t data_bits = candidate_end_pos - start_pos - 1;
        // Élargir la plage pour inclure 36230 octets (289840 bits)
        if (data_bits % 8 == 0 && data_bits >= 64 && data_bits / 8 >= 36150 && data_bits / 8 <= 36300) {
            end_pos = candidate_end_pos;
            std::cout << "Sentinelle de fin candidate à l'index " << end_pos << " (données : " << data_bits / 8 << " octets)" << std::endl;

            // Débogage : extraire et afficher les 64 premiers LSB
            std::vector<unsigned char> first_lsb;
            for (size_t j = 0; j < 64 && (start_pos + 1 + j) < modified.byteSize(); ++j) {
                first_lsb.push_back(modified.data()[start_pos + 1 + j] & 0x01);
            }
            std::cout << "Premiers 64 LSB extraits (positions " << start_pos + 1 << " à " << start_pos + 64 << ") : ";
            for (unsigned char lsb : first_lsb) {
                std::cout << (int)lsb << " ";
            }
            std::cout << std::endl;

            Image secret;
            if (cuda_naive_decrypt(modified, start_pos, end_pos, secret)) {
                std::cout << "Succès : image secrète valide extraite avec sentinelle de fin à l'index " << end_pos << std::endl;
                CUDA_CHECK(cudaFree(d_original));
                CUDA_CHECK(cudaFree(d_modified));
                CUDA_CHECK(cudaFree(d_differences));
                CUDA_CHECK(cudaFree(d_diff_count));
                return true;
            } else {
                std::cout << "Échec : les données à l'index " << end_pos << " ne forment pas une image valide." << std::endl;
            }
        } else {
            std::cout << "Différence ignorée à l'index " << candidate_end_pos << ": taille des données non alignée ou hors plage (" << data_bits << " bits)" << std::endl;
        }
    }

    std::cerr << "Erreur : aucune sentinelle de fin valide trouvée." << std::endl;
    CUDA_CHECK(cudaFree(d_original));
    CUDA_CHECK(cudaFree(d_modified));
    CUDA_CHECK(cudaFree(d_differences));
    CUDA_CHECK(cudaFree(d_diff_count));
    return false;
}

bool cuda_naive_decrypt(const Image& modified, size_t start_pos, size_t end_pos, Image& secret) {
    if (start_pos >= end_pos || end_pos > modified.byteSize()) {
        std::cerr << "Erreur : positions des sentinelles invalides." << std::endl;
        return false;
    }

    size_t data_bits = end_pos - start_pos - 1;
    std::cout << "Nombre de bits de données : " << data_bits << std::endl;
    if (data_bits % 8 != 0) {
        std::cerr << "Erreur : taille des données non alignée sur un octet." << std::endl;
        return false;
    }
    size_t secret_byte_size = data_bits / 8;
    std::cout << "Taille des données extraites (octets) : " << secret_byte_size << std::endl;

    // Allocation mémoire
    unsigned char* secret_data = new unsigned char[secret_byte_size]();
    unsigned char* d_modified;
    unsigned char* d_secret_data;
    CUDA_CHECK(cudaMalloc(&d_modified, modified.byteSize() * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_secret_data, secret_byte_size * sizeof(unsigned char)));

    // Initialiser secret_data à 0
    CUDA_CHECK(cudaMemset(d_secret_data, 0, secret_byte_size * sizeof(unsigned char)));

    // Copier les données
    CUDA_CHECK(cudaMemcpy(d_modified, modified.data(), modified.byteSize() * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Lancer le noyau avec mémoire partagée
    int threadsPerBlock = 256;
    int blocks = (data_bits + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(unsigned char);
    cuda_decrypt_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_modified, start_pos, data_bits, d_secret_data);
    CUDA_CHECK(cudaDeviceSynchronize()); // Attendre que le noyau soit terminé

    // Copier les données extraites
    CUDA_CHECK(cudaMemcpy(secret_data, d_secret_data, secret_byte_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Débogage : afficher les 64 premiers LSB utilisés
    std::cout << "Premiers 64 LSB utilisés dans cuda_decrypt_kernel : ";
    for (size_t i = 0; i < 64 && i < data_bits; ++i) {
        size_t pos = start_pos + 1 + i;
        std::cout << (int)(modified.data()[pos] & 0x01) << " ";
    }
    std::cout << std::endl;

    // Débogage : afficher les octets extraits
    std::cout << "Données extraites pour end_pos=" << end_pos << " (premiers 8 octets) : ";
    for (size_t i = 0; i < std::min<size_t>(8, secret_byte_size); ++i) {
        std::cout << std::hex << (int)secret_data[i] << " ";
    }
    std::cout << std::dec << std::endl;

    // Sauvegarder pour analyse
    std::ofstream out("extracted_data.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(secret_data), secret_byte_size);
    out.close();

    bool success = secret.createFromRawImage(secret_data, secret_byte_size);
    delete[] secret_data;
    CUDA_CHECK(cudaFree(d_modified));
    CUDA_CHECK(cudaFree(d_secret_data));

    if (!success) {
        std::cerr << "Erreur : impossible de reconstruire l'image secrète." << std::endl;
        return false;
    }

    return true;
}

int main() {
    Image original, modified, secret;

    // Extraire
    if (!original.load("../images/agile.png")) {
        std::cerr << "Erreur : impossible de charger normal" << std::endl;
        return 1;
    }
    if (!modified.load("../images/stego/agile_stego.png")) {
        std::cerr << "Erreur : impossible de charger image stego" << std::endl;
        return 1;
    }

    size_t start_pos, found_end_pos;
    if (!cuda_naive_find_sentinels(original, modified, start_pos, found_end_pos)) {
        std::cerr << "Erreur lors de la recherche des sentinelles." << std::endl;
        return 1;
    }

    if (!secret.save("../images/extracted/image_secrete_extraiteeeee.png")) {
        std::cerr << "Erreur : impossible de sauvegarder l'image extraite." << std::endl;
        return 1;
    }

    std::cout << "Extraction réussie. Image sauvegardée dans ../images/extracted/image_secrete_extraiteeeee.png" << std::endl;
    return 0;
}

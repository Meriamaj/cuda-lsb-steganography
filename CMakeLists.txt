cmake_minimum_required(VERSION 3.18)
project(SteganographyCUDA LANGUAGES CXX CUDA)

# Définir les standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Ajouter les fichiers sources
set(SOURCES
    steganography_cuda.cu
    utils/image.cpp
    utils/raw_image_loader.cpp
    utils/commonCUDA.hpp
    utils/chronoGPU.cu
    utils/chronoGPU.hpp
)

# Ajouter les dossiers d'en-tête
include_directories(
    ${CMAKE_SOURCE_DIR}/utils
)

# Trouver les bibliothèques nécessaires
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

# Trouver CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Créer l'exécutable
add_executable(steganography_cuda ${SOURCES})

# Spécifier les propriétés CUDA
set_target_properties(steganography_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 50 # Ajuster selon votre GPU (par exemple, 50 pour Maxwell, 75 pour Turing)
)

# Lier les bibliothèques
target_link_libraries(steganography_cuda
    ${PNG_LIBRARY}
    ${CUDA_LIBRARIES}
)

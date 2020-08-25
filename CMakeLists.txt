cmake_minimum_required(VERSION 3.17)
project("tensor-coin LANGUAGES CXX CUDA")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_HASH_FILES src/hash/runtime_detection.cuh src/hash/runtime_detection.cu
        src/hash/cudahash.hpp src/hash/cudahash.cpp
        src/hash/cudamultiply.cuh src/hash/cudamultiply.cu)
set(HASH_FILES src/hash/uhash.hpp src/hash/uhash.cpp src/hash/cpuhash.hpp
        src/hash/cpuhash.cpp)
set(BLOCKCHAIN_FILES src/blockchain/block.hpp src/blockchain/block.cpp
        src/blockchain/chain.hpp src/blockchain/chain.cpp
        src/blockchain/auth_wallet.cpp src/blockchain/auth_wallet.hpp)
set(TEST_FILES test/main.cpp test/tests.hpp test/tests.cpp)

set(CPU_FILES ${TEST_FILES} ${BLOCKCHAIN_FILES} ${HASH_FILES})

find_package(CUDA)
if (CUDA_FOUND)
    enable_language(CUDA)
    add_executable(tensor-coin ${CPU_FILES} ${CUDA_HASH_FILES})
    target_compile_definitions(tensor-coin PRIVATE COMPILE_WITH_CUDA)
    set_target_properties(tensor-coin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(tensor-coin -lcublas) # Works on Arch linux/CUDA11
else ()
    add_executable(tensor-coin ${CPU_FILES} ${HASH_FILES})
endif ()

find_package(OpenSSL REQUIRED)
target_link_libraries(tensor-coin OpenSSL::SSL)

cmake_minimum_required(VERSION 3.17)
project("tensor-coin LANGUAGES CUDA CXX")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_HASH_FILES src/hash/cudahash.hpp src/hash/cudahash.cpp
        src/hash/cudamultiply.cuh src/hash/cudamultiply.cu)
set(HASH_FILES src/hash/uhash.hpp src/hash/uhash.cpp src/hash/cpuhash.hpp
        src/hash/cpuhash.cpp)
set(BLOCKCHAIN_FILES src/blockchain/block.hpp src/blockchain/block.cpp
        src/blockchain/chain.hpp src/blockchain/chain.cpp
        src/blockchain/auth_wallet.cpp src/blockchain/auth_wallet.hpp)
set(TEST_FILES test/main.cpp test/tests.hpp test/tests.cpp)
add_executable(tensor-coin ${TEST_FILES} ${BLOCKCHAIN_FILES} ${HASH_FILES})
set_target_properties(tensor-coin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
find_package(OpenSSL REQUIRED)
target_link_libraries(tensor-coin OpenSSL::SSL)
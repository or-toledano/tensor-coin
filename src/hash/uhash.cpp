// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#include "uhash.hpp"
#include "cpuhash.hpp"
#include <iomanip>
#include <cstring>
#include <functional>
#include <algorithm>
#include <sstream>

#ifdef COMPILE_WITH_CUDA

#include "runtime_detection.cuh"
#include "cudahash.hpp"

#endif


using namespace tensorcoin::hash;


std::unique_ptr<UHash> UHash::make_uhash() {
#ifdef COMPILE_WITH_CUDA
    if (!detect_cuda()) return std::make_unique<CUDAHash>(CUDAHash());
#endif
    return std::make_unique<CPUHash>(CPUHash());
}

string UHash::digest_to_string(const unsigned char *src) {
//    std::ostringstream ss;
//    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i)
//        ss << std::hex << std::setw(2) << std::setfill('0')
//           << static_cast<int>(src[i]);
//    return ss.str();;
    char str[SHA256_DIGEST_LENGTH * 2 + 1];
    str[SHA256_DIGEST_LENGTH * 2] = 0;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
        std::sprintf(&str[i * 2], "%02x", (unsigned int) src[i]);
    return string(str);
}

// Split the sha256 of input (which is 64 hex letters) into 4 4x4 MATS,
// In each round, perform multiplications, combine results and use sha256
void UHash::tensor_hash(const unsigned char *src, unsigned char *dst,
                        size_t size) const {
    unsigned char original_[CUBE_SIZE]; // Before matrix multiplication
    unsigned char result_[CUBE_SIZE];  // After matrix multiplication

    unsigned char *original = original_;
    unsigned char *result = result_;
    // Here '|' denotes concatenation
    // original: [sha256(src) | garbage]
    sha256(src, original, size);
    for (int _ = 0; _ < MAT_MULT_ITERS; ++_) {
        sha256(original, original + SHA256_DIGEST_LENGTH,
               SHA256_DIGEST_LENGTH);
        // original: [sha256(src) | sha256(sha256(src))]
        memcpy(result, original, CUBE_SIZE);
        // [M0 M1 M2 M3] ==> [M3*M1+M0 M0*M2+M1 M1*M3+M2 M2*M0+M3]
        for (int i = 0; i < MATS; ++i)
            multiply_add(original, result, (i - 1 + MATS) % MATS,
                         (i + 1 + MATS) % MATS, i);
        std::swap(original, result);
        // [first half | second half] := [M3*M1+M0 M0*M2+M1 M1*M3+M2 M2*M0+M3]
        // [first half | second half] ==> [first | second ^ first]
        std::transform(original, original + SHA256_DIGEST_LENGTH,
                       original + SHA256_DIGEST_LENGTH,
                       original + SHA256_DIGEST_LENGTH,
                       std::bit_xor<>());
        // ==> [sha256(second ^ first) | second ^ first]
        sha256(original + SHA256_DIGEST_LENGTH, original,
               SHA256_DIGEST_LENGTH);
    }
    // dst = sha256(second ^ first)
    memcpy(dst, original, SHA256_DIGEST_LENGTH);

}


void
UHash::sha256(const unsigned char *src, unsigned char *dst, size_t size) const {
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, src, size);
    SHA256_Final(dst, &sha256);
}

string UHash::hash(const string &source) const {
    unsigned char dst[SHA256_DIGEST_LENGTH];
    tensor_hash(reinterpret_cast<const unsigned char *>(source.c_str()), dst,
                source.size());
    return digest_to_string(dst);
}


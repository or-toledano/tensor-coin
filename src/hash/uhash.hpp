// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
// Unified hash (name inspired by cv2::UMat, but it works differently!)
// Provides user with a unified, device (GPU/CPU) independent abstract interface
// To be inherited by device specific implementations
#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <openssl/sha.h>

#define MATS 4
#define ROWS 4
#define COLUMNS ROWS
#define DIM ROWS
// Column major (transposed) multiplication is ok in C for small dimensions
// (cache-misses-wise), and it matches the column major CUDA (which can be
// transposed but it is easier that way)
#define IDX(matrix, row, column)                                               \
    (ROWS * COLUMNS * (matrix) + ROWS * (column) + (row))
#define MAT_MULT_ITERS 1
#define MAT_SIZE (ROWS*COLUMNS)
#define CUBE_SIZE (MATS*ROWS*COLUMNS)

using std::string;
namespace tensorcoin::hash {
    class UHash {
    private:
        // Hash output (integers) to hex string
        static string digest_to_string(const unsigned char *src);

        // (bits in byte / bits in hex) * bytes in digest
        const int hex_digits = (8 / 4) * SHA256_DIGEST_LENGTH; // 64

        // Multiply the ind0 matrix with the ind1 matrix, add to the ind_res
        // matrix. Assume all indices are different and matrices are initialized
        virtual void
        multiply_add(const unsigned char *cube_src, unsigned char *cube_dst,
                     int ind0, int ind1, int ind_res) const = 0;

        virtual void
        tensor_hash(const unsigned char *src, unsigned char *dst,
                    size_t size) const;

    protected:
        virtual void
        sha256(const unsigned char *src, unsigned char *dst, size_t size) const;

    public:
        // Factory which picks the best subclass (GPU/CPU) by runtime checks
        [[nodiscard]] static std::unique_ptr<UHash> make_uhash();

        virtual ~UHash() = default;

        // The actual hash interface
        [[nodiscard]] string hash(const string &source) const;

        [[nodiscard]] int getHexDigits() const { return hex_digits; }
    };
}


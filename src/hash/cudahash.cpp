// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#include "cudahash.hpp"
#include "cudamultiply.cuh"

using namespace tensorcoin::hash;



using namespace tensorcoin::hash;
using std::string;


void
CUDAHash::multiply_add(const unsigned char *cube_src, unsigned char *cube_dst,
                       int ind0, int ind1, int ind_res) const {
    return cudaMultiplyAdd(cube_src, cube_dst, ind0, ind1, ind_res);
}


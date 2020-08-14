// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#include "cpuhash.hpp"


using namespace tensorcoin::hash;
using std::string;


void
CPUHash::multiply_add(const unsigned char *cube_src, unsigned char *cube_dst,
                      int ind0, int ind1, int ind_res) const {
    for (int i = 0; i < ROWS; ++i)           // Rows of the first matrix
        for (int j = 0; j < COLUMNS; ++j) { // Columns of the second matrix
            for (int k = 0; k < ROWS; ++k) { // Cols of first/Rows of second
                cube_dst[IDX(ind_res, i, j)] +=
                    cube_src[IDX(ind0, i, k)] * cube_src[IDX(ind1, k, j)];
            }
        }
}


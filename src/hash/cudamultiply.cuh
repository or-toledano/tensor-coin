// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#pragma once

void
cuda_multiply_add(const unsigned char *cube_src, unsigned char *cube_dst,
                  int ind0, int ind1, int ind_res);
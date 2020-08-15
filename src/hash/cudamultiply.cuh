// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#pragma once

void
cudaMultiplyAdd(const unsigned char *cubeSrc, unsigned char *cubeDst,
                int ind0, int ind1, int indRes);
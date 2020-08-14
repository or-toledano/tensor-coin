// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
// This module is included iff cuda.h exists, and checks if CUDA is available
#pragma once
#include "uhash.hpp"
using namespace tensorcoin::hash;
std::unique_ptr<UHash> make_uhash_runtime();
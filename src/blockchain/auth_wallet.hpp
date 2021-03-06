// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#pragma once

#include "chain.hpp"

using namespace tensorcoin::blockchain;
namespace tensorcoin::blockchain {
    // TODO: make this Chain implementation independent, e.i.:
    // make a chain specification, so this can receive a chain JSON
    // or any kind of serialized data, then construct a Chain in order to
    // verify it
    class AuthWallet {
    private:
        const Chain *currentChain;         // Maintain the longest valid chain
    public:
        // Was the chain mined properly
        static bool isValidChain(const Chain &chain);

        explicit AuthWallet(const Chain &chain);

        void
        updateBestChain(const Chain &newChain); // Update to the longest valid
    };
}



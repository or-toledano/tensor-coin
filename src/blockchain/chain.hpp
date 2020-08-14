// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#pragma once

#include "block.hpp"
#include "../hash/uhash.hpp"
#include <iostream>
#include <list>
#include <memory>

using std::list;
using namespace tensorcoin::hash;
namespace tensorcoin::blockchain {
    class Chain {
    private:
        const int init_target;
        int target; // Number of required zeros at the start of the hash, >=1
        const int blocksToIncTarget; // After blocksToIncTarget>=1 blocks we
        // should increase the target by 1
        const int version = 4;
        list<Block> blockList; // Mined blocks
        // Hardware/implementation polymorphism
        std::unique_ptr<UHash> uHash = UHash::make_uhash();

        friend std::ostream &operator<<(std::ostream &os, const Chain &c);

        friend class AuthWallet;


    public:
        // Does the string start with target leading zeros
        static bool isGoldenHash(const string &s, int target) noexcept;

        Chain(int target, int blocksToIncTarget);

        int getTarget() const;

        int getBlocksToChangeTarget() const;

        void mineAddBlock(string data, string claimer);

        // Adds a block to the chain, WITHOUT verifying it
        void addBlock(Block &&b);

        size_t size() const;

        // Use the chain-specific hash on the block
        string hashBlock(Block &s) const;
    };

}

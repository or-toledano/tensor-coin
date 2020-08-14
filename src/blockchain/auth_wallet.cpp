// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#include "auth_wallet.hpp"
#include "block.hpp"
#include <algorithm>

using namespace tensorcoin::blockchain;

bool AuthWallet::isValidChain(Chain *chain) {
    if (!chain->size()) return false;
    Block block = chain->blockList.front();
    string prev_hash = chain->hashBlock(block);
    auto prev_time = block.getTimeStamp();
    auto target = chain->init_target;
    int mod = 0;
    for (auto b = std::next(chain->blockList.begin());
         b != chain->blockList.end(); ++b) {
        if ((++mod) == chain->blocksToIncTarget and
            target < chain->uHash->getHexDigits()) {
            ++target;
            mod = 0;
        }
        bool tmp = target != b->getTarget() or b->getPrevHash() != prev_hash or
                   prev_time > b->getTimeStamp();
        prev_hash = chain->hashBlock(*b);
        if (tmp or !Chain::isGoldenHash(prev_hash, target)) return false;
        prev_time = b->getTimeStamp();
    }
    return true;
}

AuthWallet::AuthWallet(Chain *chain) : currentChain(chain) {
}

void AuthWallet::updateBestChain(Chain *newChain) {
    if (isValidChain(newChain) and newChain->size() > currentChain->size())
        currentChain = newChain;
}
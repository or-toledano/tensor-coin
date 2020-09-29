// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#include "chain.hpp"

using namespace tensorcoin::blockchain;


bool Chain::isGoldenHash(const string &s, int target) noexcept {
    // Note: string::npos >= target
    return s.find_first_not_of('0') >= target;
}

Chain::Chain(int target, int blocksToIncTarget)
    : init_target(target), target(target),
      blocksToIncTarget(blocksToIncTarget) {
    mineAddBlock("Genesis", "Creator");
}

int Chain::getTarget() const { return target; }

int Chain::getBlocksToChangeTarget() const { return blocksToIncTarget; }

void Chain::mineAddBlock(string data, string claimer) {
    Block b(target, Chain::version,
    (blockList.empty() ? "null" : hashBlock(blockList.back())),
        std::move(data), std::move(claimer));
    // Search for nonce which yields this.target leading zeros
    while (!isGoldenHash(hashBlock(b), target)) [[likely]] b.incNonce();
    blockList.emplace_back(std::move(b));
    // Increase the target once in a <blocksToIncTarget> times
    if (blockList.size() % blocksToIncTarget == 0 and
        target < uHash->getHexDigits())
        [[unlikely]]
        ++target;

}

size_t Chain::size() const { return blockList.size(); }

string Chain::hashBlock(const Block &s) const {
    return uHash->hash(s.stringToHash());
};

namespace tensorcoin::blockchain {
    std::ostream &operator<<(std::ostream &os, const Chain &c) {
        os << "=== *** tensorcoin Chain start *** ===\n\n";
        for (const Block &b : c.blockList) os << b << '\n';
        os << "=== *** tensorcoin Chain end *** ===\n";
        return os;
    }
}

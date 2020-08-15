// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#include "tests.hpp"
#include "../src/blockchain/block.hpp"
#include "../src/blockchain/chain.hpp"
#include "../src/blockchain/auth_wallet.hpp"
#include <iostream>

using std::cout;
using std::string;
using namespace tensorcoin::test;
using namespace tensorcoin::hash;
using namespace tensorcoin::blockchain;

void Tests::testHash() {
    string input = "When Mr. Bilbo Baggins of Bag End announced";
    std::unique_ptr<UHash> uh = UHash::make_uhash();
    cout << "Hash: <" << uh->hash(input) << ">\n";
}

void Tests::testBlock() {
    Block b(1, 4, "null", "Hey", "Orto");
    cout << b << "\n";
}

void Tests::testChain() {
    Chain c(2, 1);
    c.mineAddBlock("Just mined this block, reward me with 10 tensorCoins",
                   "Or");
    c.mineAddBlock("Give me 10 tensorCoins, and transfer 5 coins to Or",
                   "Toledano");
    cout << c << "\n";
    cout << "size:" << c.size();
}


void Tests::testValid() {
    Chain c(1, 5);
    c.mineAddBlock("Just mined this block, reward me with 10 tensorCoins",
                   "Or");
    c.mineAddBlock("Give me 10 tensorCoins, and transfer 5 coins to Or",
                   "Toledano");
    c.mineAddBlock("More", "Toledano");
    c.mineAddBlock("Od", "Toledano");
    c.mineAddBlock("50", "Toledano");
    cout << "isValidChain: " << (AuthWallet::isValidChain(&c) ? "Yes" : "No")
         << "\n";
}

void Tests::testInvalid() {
    Chain c(1, 5);
    c.mineAddBlock("Just mined this block, reward me with 10 tensorCoins",
                   "Or");
    c.mineAddBlock("Give me 10 tensorCoins, and transfer 5 coins to Or",
                   "Toledano");
    c.mineAddBlock("More", "Toledano");
    c.addBlock(Block(1, 4, "not prev_hash",
                     "My block now!", "Orto"));
    cout << "isValidChain: " << (AuthWallet::isValidChain(&c) ? "Yes" : "No")
         << "\n";
}

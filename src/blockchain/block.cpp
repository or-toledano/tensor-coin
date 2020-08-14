// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#include "block.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <ctime>
#include <vector>

using std::chrono::system_clock;
using namespace tensorcoin::blockchain;

Block::Block(int target, int version, string prev_hash, string data,
             string claimer)
    : target(target),
      timestamp(system_clock::to_time_t(system_clock::now())),
      version(version), prev_hash(std::move(prev_hash)), data(std::move(data)),
      claimer(std::move(claimer)) {}

int Block::getTarget() const { return target; }

int Block::getNonce() const { return nonce; }

void Block::incNonce() { ++nonce; }

time_t Block::getTimeStamp() const { return timestamp; }

int Block::getVersion() const { return version; }

const string &Block::getPrevHash() const { return prev_hash; }

const string &Block::getData() const { return data; }

const string &Block::getClaimer() const { return claimer; }


string Block::stringToHash() const {
    std::vector<string> strings = {std::to_string(nonce),
                                   std::to_string(target),
                                   std::to_string(timestamp),
                                   std::to_string(version),
                                   prev_hash,
                                   data,
                                   claimer};
    std::string cat;
    cat.reserve(std::accumulate(strings.begin(), strings.end(), 0,
                                [](int sum, string &s) {
                                    return sum + s.length();
                                }));
    for (string &s : strings) cat += s;
    return cat;
}

namespace tensorcoin::blockchain {
    std::ostream &operator<<(std::ostream &os, const Block &b) {
        os << "*** Tensor Coin Block start ***\n\nversion: " << b.version
           << "\nprevious hash: " << b.prev_hash << "\ntimestamp: "
           << std::put_time(std::localtime(&b.timestamp), "%F %T%z")
           << "\nrequired target (no. of leading zeros): " << b.target
           << "\nnonce: " << b.nonce << "\ntransactions: " << b.data
           << "\nBlock claimed by: " << b.claimer
           << "\n*** Tensor Coin Block end ***\n";
        return os;
    }

}

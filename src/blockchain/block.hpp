// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#pragma once

#include <string>

using std::string;
namespace tensorcoin::blockchain {
    class Block {
    private:
        int nonce = 0; // Changes to the hash, until we find a golden nonce
        const int target; // No. of leading zeros required, higher is harder
        // (note that this differs from bitcoin's target definition)
        const time_t timestamp;
        const int version;
        const string prev_hash; // Hash of the previous mined Block
        string data;
        const string claimer;

        friend std::ostream &operator<<(std::ostream &os, const Block &b);

    public:
        Block(int target, int version, string prev_hash, string data,
              string claimer);

        [[nodiscard]] int getVersion() const;

        [[nodiscard]] const string &getPrevHash() const;

        [[nodiscard]] time_t getTimeStamp() const;

        [[nodiscard]] int getTarget() const;

        [[nodiscard]] int getNonce() const;

        void incNonce();

        [[nodiscard]] const string &getData() const;

        [[nodiscard]] const string &getClaimer() const;

        // Concatenation - to be used when finding a golden nonce. Timestamp
        // is used to prevent a replay attack, and the owner is included in
        // the hash to make the proof of work credit verifiable.
        [[nodiscard]] string stringToHash() const;

    };
}


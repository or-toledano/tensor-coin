// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#pragma once
namespace tensorcoin::test {
class Tests {
  public:
    static void testHash();
    static void testBlock();
    static void testChain();
    static void testValid();
    static void testInvalid();
};
}
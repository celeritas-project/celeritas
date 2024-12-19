//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Bitset.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Bitset.hh"

#include <climits>

#include "corecel/Types.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

template<size_type N>
void test_bitset()
{
    using reference = typename Bitset<N>::reference;
    Bitset<N> x;
    EXPECT_TRUE(x.none());
    EXPECT_FALSE(x.any());
    EXPECT_FALSE(x.all());
    EXPECT_EQ(x.size(), N);

    x[N - 1] = true;
    EXPECT_EQ(x.count(), 1);
    EXPECT_FALSE(x.none());
    EXPECT_TRUE(x.any());
    EXPECT_FALSE(x.all());
    EXPECT_TRUE(x[N - 1]);

    x.flip(N - 2);
    EXPECT_TRUE(x[N - 2]);
    EXPECT_EQ(x.count(), 2);

    x.flip(N - 2);
    EXPECT_FALSE(x[N - 2]);

    x.flip(N - 1);
    x.flip();
    EXPECT_TRUE(x.all());
    EXPECT_EQ(x.count(), N);

    x.reset();
    EXPECT_EQ(x.count(), 0);
    x.set(N - 1);
    EXPECT_EQ(x.count(), 1);
    EXPECT_TRUE(x.any());
    EXPECT_TRUE(x[N - 1]);

    x.reset();

    x.set();
    EXPECT_TRUE(x.all());
    EXPECT_EQ(x.count(), N);

    x.reset();

    x[N - 2] = true;
    EXPECT_TRUE(x[N - 2]);
    EXPECT_FALSE(~x[N - 2]);

    reference r = x[N - 2];
    bool b = x[N - 2];
    EXPECT_TRUE(r);
    EXPECT_TRUE(b);
    r = false;
    EXPECT_FALSE(x[N - 2]);
    EXPECT_FALSE(r);
    EXPECT_TRUE(b);

    x[N - 2] = ~x[N - 2];

    x[N - 1] = x[N - 2];
    EXPECT_TRUE(x[N - 1]);
    x[N - 1].flip();
    EXPECT_FALSE(x[N - 1]);
    x[N - 1].flip();
    EXPECT_TRUE(x[N - 1]);
    x[N - 1] = x[N - 1];
    EXPECT_TRUE(x[N - 1]);
    x.reset(N - 1);
    EXPECT_FALSE(x[N - 1]);

    Bitset<N> y;
    EXPECT_NE(x, y);
    y = x;
    EXPECT_EQ(x, y);

    x.reset();
    y.reset();
    x[N - 1] = true;
    x[N - 3] = true;
    y[N - 2] = true;
    y[N - 3] = true;
    x ^= y;
    EXPECT_EQ(x.count(), 2);
    EXPECT_TRUE(x[N - 2]);
    EXPECT_FALSE(x[N - 3]);

    x &= y;
    EXPECT_EQ(x.count(), 1);
    EXPECT_TRUE(x[N - 2]);

    x |= y;
    EXPECT_EQ(x.count(), 2);
    EXPECT_TRUE(x[N - 2]);
    EXPECT_TRUE(x[N - 3]);

    x = ~x;
    EXPECT_EQ(x.count(), N - 2);
    EXPECT_TRUE(x[N - 1]);
    EXPECT_FALSE(x[N - 2]);
    EXPECT_FALSE(x[N - 3]);

    using flags = typename Bitset<N>::word_type;
    flags init{0};
    init = ~init;
    Bitset<N> z(init);
    size_type bits_per_word = CHAR_BIT * sizeof(typename Bitset<N>::word_type);
    EXPECT_EQ(z.count(), std::min(bits_per_word, N));
}

TEST(BitsetTest, all)
{
    test_bitset<3>();
    test_bitset<16>();
    test_bitset<32>();
    test_bitset<48>();
    test_bitset<64>();
    test_bitset<65>();
    test_bitset<66>();
}

}  // namespace test
}  // namespace celeritas

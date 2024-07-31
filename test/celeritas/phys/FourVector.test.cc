//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/FourVector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/phys/FourVector.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(FourVectorTest, basic)
{
    // Test addition
    FourVector a{{1, 2, 3}, 1000};
    FourVector b{{4, 5, 6}, 2000};
    FourVector c = a + b;

    Real3 expected_mom = {5, 7, 9};
    EXPECT_DOUBLE_EQ(a.energy + b.energy, c.energy);
    EXPECT_DOUBLE_EQ(3000, c.energy);
    EXPECT_EQ(a.mom + b.mom, c.mom);
    EXPECT_EQ(expected_mom, c.mom);

    // Test norm
    EXPECT_DOUBLE_EQ(std::sqrt(ipow<2>(c.energy) - dot_product(c.mom, c.mom)),
                     norm(c));
    EXPECT_SOFT_EQ(2999.9741665554388, norm(c));

    // Test boost_vector
    Real3 expected_boost_vector = {0.001, 0.002, 0.003};
    EXPECT_VEC_SOFT_EQ(a.mom / a.energy, boost_vector(a));
    EXPECT_VEC_SOFT_EQ(expected_boost_vector, boost_vector(a));

    // Test boost
    boost(boost_vector(a), &b);
    Real3 expected_boosted_mom
        = {6.0000300003150038, 9.0000600006300076, 12.000090000945011};
    EXPECT_VEC_SOFT_EQ(expected_boosted_mom, b.mom);
    EXPECT_SOFT_EQ(2000.0460003710041, b.energy);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

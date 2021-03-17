//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBergerReader.test.cc
//---------------------------------------------------------------------------//
#include "io/SeltzerBergerReader.hh"

#include "celeritas_test.hh"

using celeritas::SeltzerBergerReader;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SeltzerBergerReaderTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SeltzerBergerReaderTest, read)
{
    const double log_incident_energy[]
        = {-6.9078,  -6.5023,  -6.2146,  -5.8091, -5.5215, -5.2983, -5.116,
           -4.8283,  -4.6052,  -4.1997,  -3.912,  -3.5066, -3.2189, -2.9957,
           -2.8134,  -2.5257,  -2.3026,  -1.8971, -1.6094, -1.204,  -0.91629,
           -0.69315, -0.51083, -0.22314, 0,       0.40547, 0.69315, 1.0986,
           1.3863,   1.6094,   1.7918,   2.0794,  2.3026,  2.7081,  2.9957,
           3.4012,   3.6889,   3.912,    4.0943,  4.382,   4.6052,  5.0106,
           5.2983,   5.7038,   5.9915,   6.2146,  6.3969,  6.6846,  6.9078,
           7.3132,   7.6009,   8.0064,   8.294,   8.5172,  8.6995,  8.9872,
           9.2103};
    const double exiting_efrac[]
        = {1e-12, 0.025, 0.05,  0.075,  0.1,    0.15,    0.2,     0.25,
           0.3,   0.35,  0.4,   0.45,   0.5,    0.55,    0.6,     0.65,
           0.7,   0.75,  0.8,   0.85,   0.9,    0.925,   0.95,    0.97,
           0.99,  0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 1};

    SeltzerBergerReader reader;
    {
        const auto result = reader(1); // Hydrogen
        EXPECT_VEC_SOFT_EQ(log_incident_energy, result.x);
        EXPECT_VEC_SOFT_EQ(exiting_efrac, result.y);
        EXPECT_EQ(1824, result.value.size());
        EXPECT_EQ(4.6875e-2, result.value.back());
    }
    {
        const auto result = reader(94); // Plutonium
        EXPECT_VEC_SOFT_EQ(log_incident_energy, result.x);
        EXPECT_VEC_SOFT_EQ(exiting_efrac, result.y);
        EXPECT_EQ(1824, result.value.size());
        EXPECT_EQ(1.50879, result.value.back());
    }
}

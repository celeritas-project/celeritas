//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Constants.test.cc
//---------------------------------------------------------------------------//
#include "base/Constants.hh"
#include "base/Units.hh"

#include "celeritas_test.hh"

using namespace celeritas::units;
using namespace celeritas::constants;
using celeritas::real_type;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(UnitsTest, equivalence)
{
    EXPECT_DOUBLE_EQ(ampere * ampere * second * second * second * second
                         / (kilogram * meter * meter),
                     farad);
    EXPECT_DOUBLE_EQ(kilogram * meter * meter / (second * second), joule);

    constexpr real_type erg = gram * centimeter * centimeter
                              / (second * second);

    EXPECT_EQ(real_type(1), erg);
    EXPECT_EQ(1e7 * erg, joule);
}

//! Test that no precision is lost for cm<->m and other integer factors.
TEST(UnitsTest, exact_equivalence)
{
    EXPECT_EQ(299792458e2, c_light);     // cm/s
    EXPECT_EQ(6.62607015e-27, h_planck); // erg
}

TEST(ConstantsTest, formulas)
{
    EXPECT_SOFT_NEAR(e_electron * e_electron
                         / (2 * alpha_fine_structure * h_planck * c_light),
                     eps_electric,
                     1e-11);
    EXPECT_SOFT_NEAR(
        1 / (eps_electric * c_light * c_light), mu_magnetic, 1e-11);
    EXPECT_SOFT_NEAR(
        hbar_planck / (alpha_fine_structure * electron_mass * c_light),
        a0_bohr,
        1e-11);
    EXPECT_SOFT_NEAR(alpha_fine_structure * alpha_fine_structure * a0_bohr,
                     r_electron,
                     1e-11);
}

TEST(ConstantsTest, derivative)
{
    // Compared against definition of Dalton, table 8 of SI 2019
    EXPECT_SOFT_NEAR(1.66053906660e-27 * kilogram, atomic_mass, 1e-11);
    EXPECT_SOFT_NEAR(1.602176634e-19, e_electron * volt, 1e-11);

    // CODATA 2018 listings
    EXPECT_SOFT_NEAR(
        1.49241808560e-10 * joule, atomic_mass * c_light * c_light, 1e-11);
    EXPECT_SOFT_NEAR(931.49410242e6 * e_electron * volt,
                     atomic_mass * c_light * c_light,
                     1e-11);
}

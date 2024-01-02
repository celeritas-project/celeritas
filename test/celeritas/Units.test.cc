//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Units.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/Units.hh"

#include "celeritas/UnitTypes.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace units
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(UnitsTest, equivalence)
{
    EXPECT_REAL_EQ(ampere * ampere * second * second * second * second
                       / (kilogram * meter * meter),
                   farad);
    EXPECT_REAL_EQ(kilogram * meter * meter / (second * second), joule);

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        constexpr real_type erg = gram * centimeter * centimeter
                                  / (second * second);

        EXPECT_EQ(real_type(1), erg);
        EXPECT_EQ(1e7 * erg, joule);
        EXPECT_REAL_EQ(1e4, tesla);
        EXPECT_REAL_EQ(0.1, coulomb);
    }
}

//---------------------------------------------------------------------------//
TEST(UnitsTest, traits)
{
    EXPECT_EQ(CgsTraits::Length::value(), centimeter);
    EXPECT_EQ(CgsTraits::Mass::value(), gram);
    EXPECT_EQ(CgsTraits::Time::value(), second);

    EXPECT_EQ(SiTraits::Length::value(), meter);
    EXPECT_EQ(SiTraits::Mass::value(), kilogram);
    EXPECT_EQ(SiTraits::Time::value(), second);

    EXPECT_EQ(ClhepTraits::Length::value(), millimeter);
    EXPECT_EQ(ClhepTraits::Mass::value(), MevPerCsq::value());
    EXPECT_EQ(ClhepTraits::Time::value(), nanosecond);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace units
}  // namespace celeritas

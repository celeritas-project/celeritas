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
        EXPECT_REAL_EQ(1e3 * tesla, ClhepUnitBField::value());
    }
    else if (CELERITAS_UNITS == CELERITAS_UNITS_CLHEP)
    {
#if CELERITAS_UNITS == CELERITAS_UNITS_CLHEP
        // e_electron is *not* a unit in anything but CLHEP
        EXPECT_REAL_EQ(constants::e_electron, units::e_electron);
#endif
        EXPECT_EQ(1, constants::e_electron);
        EXPECT_REAL_EQ(1e6 / 1.602176634e-19, kilogram);
        EXPECT_REAL_EQ(1e-3, tesla);
    }
}

//---------------------------------------------------------------------------//
TEST(UnitsTest, traits)
{
    EXPECT_EQ(NativeTraits::Length::value(), 1);
    EXPECT_EQ(NativeTraits::Mass::value(), 1);
    EXPECT_EQ(NativeTraits::Time::value(), 1);
    EXPECT_EQ(NativeTraits::BField::value(), 1);

    EXPECT_REAL_EQ(CgsTraits::Length::value(), centimeter);
    EXPECT_REAL_EQ(CgsTraits::Mass::value(), gram);
    EXPECT_REAL_EQ(CgsTraits::Time::value(), second);
    EXPECT_REAL_EQ(CgsTraits::BField::value(), gauss);

    EXPECT_REAL_EQ(SiTraits::Length::value(), meter);
    EXPECT_REAL_EQ(SiTraits::Mass::value(), kilogram);
    EXPECT_REAL_EQ(SiTraits::Time::value(), second);
    EXPECT_REAL_EQ(SiTraits::BField::value(), tesla);

    EXPECT_REAL_EQ(ClhepTraits::Length::value(), millimeter);
    EXPECT_REAL_EQ(ClhepTraits::Mass::value(), ClhepUnitMass::value());
    EXPECT_REAL_EQ(ClhepTraits::Time::value(), nanosecond);
    EXPECT_REAL_EQ(ClhepTraits::BField::value(), ClhepUnitBField::value());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace units
}  // namespace celeritas

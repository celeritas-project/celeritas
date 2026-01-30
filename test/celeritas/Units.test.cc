//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Units.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/Units.hh"

#include "celeritas/UnitTypes.hh"
#include "celeritas/ext/GeantUnits.hh"

#include "celeritas_test.hh"

#if CELERITAS_USE_GEANT4
#    include <CLHEP/Units/PhysicalConstants.h>
#    include <CLHEP/Units/SystemOfUnits.h>

#    include "geocel/g4/Convert.hh"
#endif

namespace celeritas
{
namespace units
{
namespace test
{
//---------------------------------------------------------------------------//
// Locally replace the Celeritas "real" expectation for one that forces
// Constant objects to double-precision
#undef EXPECT_REAL_EQ
#define EXPECT_REAL_EQ(a, b) \
    EXPECT_DOUBLE_EQ(static_cast<double>(a), static_cast<double>(b))

//---------------------------------------------------------------------------//
TEST(UnitsTest, equivalence)
{
    EXPECT_REAL_EQ(ampere * ampere * second * second * second * second
                       / (kilogram * meter * meter),
                   farad);
    EXPECT_REAL_EQ(kilogram * meter * meter / (second * second), joule);

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        constexpr auto erg = gram * centimeter * centimeter / (second * second);

        EXPECT_EQ(real_type(1), erg);
        EXPECT_EQ(1e7 * erg, joule);
        EXPECT_REAL_EQ(1e4, tesla);
        EXPECT_REAL_EQ(0.1, coulomb);
        EXPECT_REAL_EQ(1e3 * tesla, ClhepUnitBField::value());
    }
    else if (CELERITAS_UNITS == CELERITAS_UNITS_CLHEP)
    {
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
TEST(UnitsTest, trait_visitor)
{
    auto get_length_str = [](auto utraits) {
        using Length = typename decltype(utraits)::Length;
        return Length::label();
    };

    EXPECT_STREQ("cm", visit_unit_system(get_length_str, UnitSystem::cgs));
    EXPECT_STREQ("m", visit_unit_system(get_length_str, UnitSystem::si));
    EXPECT_STREQ("mm", visit_unit_system(get_length_str, UnitSystem::clhep));
}

//---------------------------------------------------------------------------//
TEST(UnitsTest, clhep)
{
#if CELERITAS_USE_GEANT4
    {
        EXPECT_SOFT_EQ(0.001, CLHEP::tesla);
        if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
        {
            EXPECT_SOFT_EQ(1e4, units::tesla);
        }
        else if (CELERITAS_UNITS == CELERITAS_UNITS_CLHEP)
        {
            EXPECT_SOFT_EQ(CLHEP::tesla, units::tesla);
        }

        double g4_native = 2.5 * CLHEP::tesla;
        auto celer_native = convert_from_geant(g4_native, clhep_field);
        EXPECT_SOFT_EQ(2.5 * units::tesla, celer_native);
        EXPECT_SOFT_EQ(2.5, native_value_to<FieldTesla>(celer_native).value());
        EXPECT_SOFT_EQ(g4_native, convert_to_geant(celer_native, clhep_field));
    }
    {
        double g4_native = 1.5 * CLHEP::s;
        auto celer_native = convert_from_geant(g4_native, clhep_time);
        EXPECT_SOFT_EQ(1.5 * units::second, celer_native);
        EXPECT_SOFT_EQ(g4_native, convert_to_geant(celer_native, clhep_time));
    }
    {
        double g4_native = 1.5 * CLHEP::meter;
        auto celer_native = convert_from_geant(g4_native, clhep_length);
        EXPECT_SOFT_EQ(1.5 * units::meter, celer_native);
        EXPECT_SOFT_EQ(g4_native, convert_to_geant(celer_native, clhep_length));
    }
#else
    GTEST_SKIP() << "CLHEP is not available";
#endif
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace units
}  // namespace celeritas

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
using namespace celeritas::units::literals;
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
TEST(UnitsTest, literals)
{
    EXPECT_REAL_EQ(2.5 * units::centimeter, 2.5_centimeter);
    EXPECT_REAL_EQ(2.5 * units::centimeter, 2.5_cm);
    EXPECT_REAL_EQ(3 * units::meter, 3_meter);
    EXPECT_REAL_EQ(3 * units::meter, 3_m);
    EXPECT_REAL_EQ(7 * units::millimeter, 7_millimeter);
    EXPECT_REAL_EQ(7 * units::millimeter, 7_mm);
    EXPECT_REAL_EQ(4 * units::nanosecond, 4_nanosecond);
    EXPECT_REAL_EQ(4 * units::nanosecond, 4_ns);
    EXPECT_REAL_EQ(1.25 * units::tesla, 1.25_tesla);
    EXPECT_REAL_EQ(1.25 * units::tesla, 1.25_T);
    EXPECT_REAL_EQ(2 * units::gram, 2_gram);
    EXPECT_REAL_EQ(2 * units::gram, 2_g);
    EXPECT_REAL_EQ(5 * units::barn, 5_barn);
    EXPECT_REAL_EQ(5 * units::barn, 5_b);
    EXPECT_REAL_EQ(1.5 * units::second, 1.5_second);
    EXPECT_REAL_EQ(1.5 * units::second, 1.5_s);
    EXPECT_REAL_EQ(1 * units::gauss, 1_gauss);
    EXPECT_REAL_EQ(1 * units::gauss, 1_G);
    EXPECT_REAL_EQ(300 * units::kelvin, 300_kelvin);
    EXPECT_REAL_EQ(300 * units::kelvin, 300_K);
    EXPECT_REAL_EQ(2 * units::kilogram, 2_kilogram);
    EXPECT_REAL_EQ(2 * units::kilogram, 2_kg);
    EXPECT_REAL_EQ(9 * units::newton, 9_newton);
    EXPECT_REAL_EQ(9 * units::newton, 9_N);
    EXPECT_REAL_EQ(10 * units::joule, 10_joule);
    EXPECT_REAL_EQ(10 * units::joule, 10_J);
    EXPECT_REAL_EQ(11 * units::coulomb, 11_coulomb);
    EXPECT_REAL_EQ(11 * units::coulomb, 11_C);
    EXPECT_REAL_EQ(12 * units::ampere, 12_ampere);
    EXPECT_REAL_EQ(12 * units::ampere, 12_A);
    EXPECT_REAL_EQ(13 * units::volt, 13_volt);
    EXPECT_REAL_EQ(13 * units::volt, 13_V);
    EXPECT_REAL_EQ(14 * units::farad, 14_farad);
    EXPECT_REAL_EQ(14 * units::farad, 14_F);
    EXPECT_REAL_EQ(15 * units::micrometer, 15_micrometer);
    EXPECT_REAL_EQ(15 * units::micrometer, 15_um);
    EXPECT_REAL_EQ(16 * units::nanometer, 16_nanometer);
    EXPECT_REAL_EQ(16 * units::nanometer, 16_nm);
    EXPECT_REAL_EQ(17 * units::femtometer, 17_femtometer);
    EXPECT_REAL_EQ(17 * units::femtometer, 17_fm);
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
        EXPECT_SOFT_EQ(2.5_T, celer_native);
        EXPECT_SOFT_EQ(2.5, native_value_to<FieldTesla>(celer_native).value());
        EXPECT_SOFT_EQ(g4_native, convert_to_geant(celer_native, clhep_field));
    }
    {
        double g4_native = 1.5 * CLHEP::s;
        auto celer_native = convert_from_geant(g4_native, clhep_time);
        EXPECT_SOFT_EQ(1.5_s, celer_native);
        EXPECT_SOFT_EQ(g4_native, convert_to_geant(celer_native, clhep_time));
    }
    {
        double g4_native = 1.5 * CLHEP::meter;
        auto celer_native = convert_from_geant(g4_native, clhep_length);
        EXPECT_SOFT_EQ(1.5_m, celer_native);
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

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Constants.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/Constants.hh"

#include <cmath>

#include "corecel/Config.hh"

#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"

#include "celeritas_test.hh"

#if CELERITAS_USE_GEANT4
#    include <CLHEP/Units/PhysicalConstants.h>
#    include <CLHEP/Units/SystemOfUnits.h>
#endif

namespace celeritas
{
namespace constants
{
namespace test
{
//---------------------------------------------------------------------------//
// CLHEP units introduce extra error due to repeated operations with
// non-representable values
auto constexpr clhep_tol = SoftEqual<real_type>{}.rel()
                           * (CELERITAS_UNITS == CELERITAS_UNITS_CLHEP ? 5 : 1);

TEST(ConstantsTest, mathematical)
{
    EXPECT_DOUBLE_EQ(static_cast<double>(euler), std::exp(1.0));
    EXPECT_DOUBLE_EQ(static_cast<double>(pi), std::acos(-1.0));
    EXPECT_DOUBLE_EQ(static_cast<double>(sqrt_two), std::sqrt(2.0));
    EXPECT_DOUBLE_EQ(static_cast<double>(sqrt_three), std::sqrt(3.0));
}

//! Test that no precision is lost for cm<->m and other integer factors.
TEST(ConstantsTest, exact_equivalence)
{
    using celeritas::units::centimeter;
    using celeritas::units::second;

    EXPECT_EQ(299792458e2, c_light / (centimeter / second));  // cm/s
    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        EXPECT_EQ(6.62607015e-27, h_planck);  // erg
    }
}

TEST(ConstantsTest, formulas)
{
    EXPECT_SOFT_NEAR(e_electron * e_electron
                         / (2 * alpha_fine_structure * h_planck * c_light),
                     eps_electric,
                     clhep_tol);
    EXPECT_SOFT_EQ(1 / (eps_electric * c_light * c_light), mu_magnetic);
    EXPECT_SOFT_EQ(
        hbar_planck / (alpha_fine_structure * electron_mass * c_light),
        a0_bohr);
    EXPECT_SOFT_EQ(alpha_fine_structure * alpha_fine_structure * a0_bohr,
                   r_electron);
}

TEST(ConstantsTest, clhep_codata)
{
    // Values differ from the CLHEP constants (CODATA 2006) by ~1e-7 due to
    // the 2019 change in SI using e- charge as an exact definition
    // rather than a measured constant
    constexpr Constant old_e_electron{1.602176487e-19 * units::coulomb};

    using MevMass = Quantity<units::MevPerCsq, double>;

    // Like other CODATA constants, derivative values seem to be accurate only
    // to ~1e-10 due to propagation of uncertainty
    EXPECT_SOFT_NEAR(
        native_value_to<MevMass>(codata2006::electron_mass).value(),
        0.510998910 * old_e_electron / e_electron,
        5e-10);
    EXPECT_SOFT_NEAR(native_value_to<MevMass>(codata2006::proton_mass).value(),
                     938.272013 * old_e_electron / e_electron,
                     5e-10);
}

TEST(ConstantsTest, clhep)
{
#if CELERITAS_USE_GEANT4
    EXPECT_SOFT_NEAR(
        a0_bohr / units::centimeter, CLHEP::Bohr_radius / CLHEP::cm, 1e-7);
    EXPECT_SOFT_NEAR(alpha_fine_structure, CLHEP::fine_structure_const, 1e-9);
    EXPECT_SOFT_NEAR(atomic_mass / units::gram, CLHEP::amu / CLHEP::gram, 1e-7);
    EXPECT_SOFT_NEAR(eps_electric
                         / (units::coulomb * units::coulomb
                            / (units::newton * units::meter * units::meter)),
                     CLHEP::epsilon0
                         / (CLHEP::coulomb * CLHEP::coulomb
                            / (CLHEP::newton * CLHEP::meter * CLHEP::meter)),
                     1e-7);
    EXPECT_SOFT_NEAR(h_planck, CLHEP::h_Planck, 1e-7);
    EXPECT_SOFT_NEAR(k_boltzmann, CLHEP::k_Boltzmann, 1e-7);
    EXPECT_SOFT_NEAR(
        mu_magnetic * units::ampere * units::ampere / units::newton,
        CLHEP::mu0 * CLHEP::ampere * CLHEP::ampere / CLHEP::newton,
        1e-7);
    EXPECT_SOFT_NEAR(na_avogadro, CLHEP::Avogadro, 1e-7);
    EXPECT_SOFT_NEAR(r_electron / units::centimeter,
                     CLHEP::classic_electr_radius / CLHEP::cm,
                     1e-7);
    EXPECT_SOFT_NEAR(lambdabar_electron / units::centimeter,
                     CLHEP::electron_Compton_length / CLHEP::cm,
                     1e-7);
#else
    GTEST_SKIP() << "CLHEP is not available";
#endif
}

TEST(ConstantsTest, derivative)
{
    // Compared against definition of Dalton, table 8 of SI 2019
    EXPECT_SOFT_EQ(1.66053906660e-27 * units::kilogram, atomic_mass);
    EXPECT_SOFT_EQ(1.602176634e-19 * units::joule, e_electron * units::volt);

    // CODATA 2018 listings
    EXPECT_SOFT_NEAR(1.49241808560e-10 * units::joule,
                     atomic_mass * c_light * c_light,
                     clhep_tol);
    EXPECT_SOFT_NEAR(931.49410242e6 * e_electron * units::volt,
                     atomic_mass * c_light * c_light,
                     clhep_tol);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace constants
}  // namespace celeritas

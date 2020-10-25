//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Material.test.cc
//---------------------------------------------------------------------------//
#include "physics/material/ElementView.hh"
#include "physics/material/MaterialView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "physics/material/MaterialParams.hh"
#include "physics/material/detail/Utils.hh"

#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "physics/base/Units.hh"
#include "physics/material/ElementDef.hh"
// #include "Material.test.hh"

using celeritas::ElementDef;
using celeritas::real_type;
// using namespace celeritas_test;

//---------------------------------------------------------------------------//
/*!
 * Test mass radiation coefficient calculation.
 *
 * Reference values are from
 * https://pdg.lbl.gov/2020/AtomicNuclearProperties/index.html
 *
 * We test all the special cases (H through Li) plus the regular case plus the
 * "out-of-bounds" case for transuranics.
 *
 * The mass radiation coefficient (the inverse of which is referred to as
 * "radiation length" in the PDG physics review) is an inverse length, divided
 * by the material density: units are [cm^2/g]. It's analogous to the mass
 * attenutation coefficient mu / rho.
 */
TEST(MaterialUtils, radiation_length)
{
    using celeritas::detail::calc_mass_rad_coeff;
    using celeritas::units::AmuMass;

    ElementDef el;

    // Hydrogen
    el.atomic_number = 1;
    el.atomic_mass   = AmuMass{1.008};
    EXPECT_SOFT_NEAR(63.04, 1 / calc_mass_rad_coeff(el), 1e-3);
    real_type hydrogen_density = 8.376e-05; // g/cc
    EXPECT_SOFT_NEAR(
        7.527e5, 1 / (hydrogen_density * calc_mass_rad_coeff(el)), 1e-3);

    // Helium
    el.atomic_number = 2;
    el.atomic_mass   = AmuMass{4.002602};
    EXPECT_SOFT_NEAR(94.32, 1 / calc_mass_rad_coeff(el), 1e-3);

    // Lithium
    el.atomic_number = 3;
    el.atomic_mass   = AmuMass{6.94};
    EXPECT_SOFT_NEAR(82.77, 1 / calc_mass_rad_coeff(el), 1e-3);

    // Beryllium
    el.atomic_number = 4;
    el.atomic_mass   = AmuMass{9.0121831};
    EXPECT_SOFT_NEAR(65.19, 1 / calc_mass_rad_coeff(el), 1e-3);

    // Aluminum
    el.atomic_number = 13;
    el.atomic_mass   = AmuMass{26.9815385};
    EXPECT_SOFT_NEAR(24.01, 1 / calc_mass_rad_coeff(el), 1e-3);

    // Uranium
    el.atomic_number = 92;
    el.atomic_mass   = AmuMass{238.02891};
    EXPECT_SOFT_NEAR(6.00, 1 / calc_mass_rad_coeff(el), 1e-3);

    // Plutonium-244 [NOTE: accuracy decreases compared to tabulated values]
    el.atomic_number = 94;
    el.atomic_mass   = AmuMass{244.06420};
    EXPECT_SOFT_NEAR(5.93, 1 / calc_mass_rad_coeff(el), 1e-2);
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class MaterialParamsTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MaterialParamsTest, all)
{
    // MTestInput input;
    // input.num_threads = 0;
    // auto result = m_test(input);
    // PRINT_EXPECTED(result.foo);
    // EXPECT_VEC_SOFT_EQ(expected_foo, result.foo);
}

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/CylMapMagneticField.test.cc
//---------------------------------------------------------------------------//
#include "accel/CylMapMagneticField.hh"

#include <memory>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4MagneticField.hh>

#include "corecel/grid/VectorUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/CylMapFieldInput.hh"
#include "celeritas/field/CylMapFieldParams.hh"

#include "LinearMagFieldTestBase.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class CylMapMagneticFieldTest : public LinearMagFieldTestBase
{
};

TEST_F(CylMapMagneticFieldTest, make_input)
{
    using CLHEP::cm;
    using CLHEP::deg;

    // Create cylindrical grid
    std::vector<G4double> r_grid = {0.0 * cm, 1.5 * cm};
    std::vector<G4double> phi_values = {0, 30 * deg, 180 * deg, 360 * deg};
    std::vector<G4double> z_grid = {-3 * cm, -1 * cm, 0 * cm};

    // Convert
    auto inp
        = MakeCylMapFieldInput(this->g4field(), r_grid, phi_values, z_grid);

    // Check (being careful with units)
    std::vector<real_type> field_tesla(inp.field.size());
    for (auto i : range(inp.field.size()))
    {
        field_tesla[i]
            = native_value_to<units::FieldTesla>(inp.field[i]).value();
    }

    // Field values are computed at cylindrical grid points *and* stored as
    // [r, phi, z]
    EXPECT_EQ(2 * 4 * 3 * 3, field_tesla.size());
    // clang-format off
    static double const expected_field_tesla[] = {
        -1.05, -1.65, -0.75, -1.05, -1.65, 2.25,
        -1.05, -1.65, 3.75, -1.05, -1.65, -0.75,
        -1.05, -1.65, 2.25, -1.05, -1.65, 3.75,
        1.05, 1.65, -0.75, 1.05, 1.65, 2.25,
        1.05, 1.65, 3.75, -1.05, -1.65, -0.75,
        -1.05, -1.65, 2.25, -1.05, -1.65, 3.75,
        1.2, -1.65, -0.75, 1.2, -1.65, 2.25,
        1.2, -1.65, 3.75, 0.5156733260263, -0.9039419162443, -0.75,
        0.5156733260263, -0.9039419162443, 2.25, 0.5156733260263, -0.9039419162443, 3.75,
        3.3, 1.65, -0.75, 3.3, 1.65, 2.25,
        3.3, 1.65, 3.75, 1.2, -1.65, -0.75,
        1.2, -1.65, 2.25, 1.2, -1.65, 3.75,
    };
    // clang-format on
    EXPECT_VEC_NEAR(expected_field_tesla, field_tesla, 1e-5);

    EXPECT_VEC_SOFT_EQ(
        (Real2{0.0, 1.5}),
        (Real2{to_cm(inp.grid_r.front()), to_cm(inp.grid_r.back())}));
    EXPECT_EQ(2, inp.grid_r.size());

    ASSERT_EQ(4, inp.grid_phi.size());
    EXPECT_SOFT_EQ(0, value_as<RealTurn>(inp.grid_phi.front()));
    EXPECT_SOFT_EQ(1.0 / 12, value_as<RealTurn>(inp.grid_phi[1]));
    EXPECT_SOFT_EQ(1.0, value_as<RealTurn>(inp.grid_phi.back()));

    EXPECT_VEC_SOFT_EQ(
        (Real2{-3, 0}),
        (Real2{to_cm(inp.grid_z.front()), to_cm(inp.grid_z.back())}));
    EXPECT_EQ(3, inp.grid_z.size());
}

// Test that the field mapping is roughly equivalent
TEST_F(CylMapMagneticFieldTest, geant_calculation)
{
    using CLHEP::cm;
    using CLHEP::deg;
    using CLHEP::tesla;

    // Create cylindrical grid covering the region of interest
    std::vector<G4double> r_grid = linspace(0, 2 * cm, 32);
    std::vector<G4double> phi_values = linspace(0, 360 * deg, 32);
    std::vector<G4double> z_grid = linspace(-4 * cm, 4 * cm, 128);

    // Create mapped magnetic field with cylindrical grid
    CylMapMagneticField cyl_field{std::make_shared<CylMapFieldParams>(
        MakeCylMapFieldInput(this->g4field(), r_grid, phi_values, z_grid))};

    constexpr SoftEqual tol{0.05, 0.05};

    // Check where the true value is zero
    Dbl3 pos{0.7 * cm, 1.1 * cm, -2.5 * cm};
    EXPECT_VEC_NEAR((Dbl3{0, 0, 0}), calc_field(this->g4field(), pos), 1e-6);
    this->check_field(cyl_field, pos, tol);

    // Check where the true value should be ~{0,0,1.5T}
    pos = {0.7 * cm, 1.1 * cm, -1.5 * cm};
    EXPECT_VEC_NEAR((Dbl3{0, 0, 1.5}), calc_field(this->g4field(), pos), 1e-6);
    this->check_field(cyl_field, pos, tol);

    // Check elsewhere inside cylindrical volume
    pos = {0.5 * cm, -1.6 * cm, 2.5 * cm};
    this->check_field(cyl_field, pos, tol);

    // Check outside sample volume
    pos = {1.0 * cm, 1.0 * cm, 8.0 * cm};
    EXPECT_VEC_NEAR((Dbl3{0, 0, 0}), this->calc_field(cyl_field, pos), tol);
    pos = {1.0 * cm, 1.0 * cm, -8.0 * cm};
    EXPECT_VEC_NEAR((Dbl3{0, 0, 0}), this->calc_field(cyl_field, pos), tol);
    pos = {1.9 * cm, -1.9 * cm, 0.0 * cm};
    EXPECT_VEC_NEAR((Dbl3{0, 0, 0}), this->calc_field(cyl_field, pos), tol);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

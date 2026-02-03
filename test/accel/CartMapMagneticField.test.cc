//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/CartMapMagneticField.test.cc
//---------------------------------------------------------------------------//
#include "accel/CartMapMagneticField.hh"

#include <memory>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4MagneticField.hh>

#include "geocel/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/CartMapFieldParams.hh"

#include "LinearMagFieldTestBase.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class CartMapMagneticFieldTest : public LinearMagFieldTestBase
{
};

TEST_F(CartMapMagneticFieldTest, make_input)
{
    // Convert
    auto inp = MakeCartMapFieldInput(this->g4field(), [] {
        using CLHEP::cm;

        CartMapFieldGridParams grid;
        grid.x.min = 0.1 * cm;
        grid.x.max = 2.1 * cm;
        grid.x.num = 2;

        grid.y.min = 0.5 * cm;
        grid.y.max = 1.5 * cm;
        grid.y.num = 3;

        grid.z.min = -3 * cm;
        grid.z.max = 0 * cm;
        grid.z.num = 4;

        return grid;
    }());

    // Check (being careful with units)
    std::vector<real_type> field_tesla(inp.field.size());
    for (auto i : range(inp.field.size()))
    {
        field_tesla[i]
            = native_value_to<units::FieldTesla>(inp.field[i]).value();
    }
    static double const expected_field_tesla[] = {
        -0.9, -0.9,  -0.75, -0.9, -0.9,  0.75,  -0.9, -0.9,  2.25,
        -0.9, -0.9,  3.75,  -0.9, -0.15, -0.75, -0.9, -0.15, 0.75,
        -0.9, -0.15, 2.25,  -0.9, -0.15, 3.75,  -0.9, 0.6,   -0.75,
        -0.9, 0.6,   0.75,  -0.9, 0.6,   2.25,  -0.9, 0.6,   3.75,
        2.1,  -0.9,  -0.75, 2.1,  -0.9,  0.75,  2.1,  -0.9,  2.25,
        2.1,  -0.9,  3.75,  2.1,  -0.15, -0.75, 2.1,  -0.15, 0.75,
        2.1,  -0.15, 2.25,  2.1,  -0.15, 3.75,  2.1,  0.6,   -0.75,
        2.1,  0.6,   0.75,  2.1,  0.6,   2.25,  2.1,  0.6,   3.75,
    };
    EXPECT_VEC_SOFT_EQ(expected_field_tesla, field_tesla);

    EXPECT_VEC_SOFT_EQ((Real2{0.1, 2.1}),
                       (Real2{to_cm(inp.x.min), to_cm(inp.x.max)}));
    EXPECT_EQ(2, inp.x.num);
    EXPECT_VEC_SOFT_EQ((Real2{0.5, 1.5}),
                       (Real2{to_cm(inp.y.min), to_cm(inp.y.max)}));
    EXPECT_EQ(3, inp.y.num);
    EXPECT_VEC_SOFT_EQ((Real2{-3, 0}),
                       (Real2{to_cm(inp.z.min), to_cm(inp.z.max)}));
    EXPECT_EQ(4, inp.z.num);
}

// Test that the field mapping is roughly equivalent (linear should work, but
// covfie/single precision introduce some errors)
TEST_F(CartMapMagneticFieldTest, geant_calculation)
{
    using CLHEP::cm;
    using CLHEP::tesla;

    // Create mapped magnetic field
    CartMapMagneticField cart_field{std::make_shared<CartMapFieldParams>(
        MakeCartMapFieldInput(this->g4field(), [] {
            CartMapFieldGridParams grid;
            grid.x.min = 0.0 * cm;
            grid.x.max = 1.0 * cm;
            grid.x.num = 4;

            grid.y.min = 0. * cm;
            grid.y.max = 2.0 * cm;
            grid.y.num = 8;

            grid.z.min = -4 * cm;
            grid.z.max = 0 * cm;
            grid.z.num = 16;
            return grid;
        }()))};

    constexpr SoftEqual tol{1e-5, 1e-6};
    SCOPED_TRACE("cartesian");

    // Check where the true value is zero
    Dbl3 pos{0.7 * cm, 1.1 * cm, -2.5 * cm};
    this->check_field(cart_field, pos, tol);

    // Check where the true value should be ~{0,0,1.5T}
    pos = {0.7 * cm, 1.1 * cm, -1.5 * cm};
    this->check_field(cart_field, pos, tol);

    // Check elsewhere inside box
    pos = {0.5 * cm, 0.11 * cm, -3.9 * cm};
    this->check_field(cart_field, pos, tol);

    pos = {-1 * cm, 0.1 * cm, -0.1 * cm};
    EXPECT_VEC_NEAR(
        (Dbl3{-1.05, -1.5, 3.6}), this->calc_field(cart_field, pos), tol);
    pos = {1 * cm, 0.1 * cm, -0.1 * cm};
    EXPECT_VEC_NEAR(
        (Dbl3{0.45, -1.5, 3.6}), this->calc_field(cart_field, pos), tol);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

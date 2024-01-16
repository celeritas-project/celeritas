//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/RZMapMagneticField.test.cc
//---------------------------------------------------------------------------//
#include "accel/RZMapMagneticField.hh"

#include <CLHEP/Units/SystemOfUnits.h>

#include "celeritas/ext/Convert.geant.hh"
#include "celeritas/field/RZMapField.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/field/RZMapFieldParams.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class RZMapMagneticFieldTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(RZMapMagneticFieldTest, uniform_z)
{
    std::shared_ptr<RZMapFieldParams> params = [] {
        // NOTE: RZ field map input is in native units
        RZMapFieldInput inp;
        inp.num_grid_z = 2;
        inp.num_grid_r = 2;
        inp.min_z = -10.0 * units::centimeter;
        inp.min_r = 0;
        inp.max_z = 10.0 * units::centimeter;
        inp.max_r = 7.0 * units::centimeter;
        inp.field_z.assign(4, 1.0 * units::tesla);
        inp.field_r.assign(4, 0.0);
        return std::make_shared<RZMapFieldParams>(std::move(inp));
    }();

    RZMapMagneticField g4mag_field{params};

    G4ThreeVector g4point;
    G4ThreeVector g4field_value{0, 0, 0};

    // Calculate inside the grid, comparing result *IN TESLA*
    g4point = convert_to_geant(Real3{5, 1, 2}, CLHEP::cm);
    g4mag_field.GetFieldValue(&g4point[0], &g4field_value[0]);
    EXPECT_VEC_SOFT_EQ((Real3{0, 0, 1}),
                       convert_from_geant(g4field_value, CLHEP::tesla));

    // Calculate outside the grid, comparing result *IN TESLA*
    g4point = convert_to_geant(Real3{12, 0, 0}, CLHEP::cm);
    g4mag_field.GetFieldValue(&g4point[0], &g4field_value[0]);
    EXPECT_VEC_SOFT_EQ((Real3{0, 0, 0}),
                       convert_from_geant(g4field_value, CLHEP::tesla));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

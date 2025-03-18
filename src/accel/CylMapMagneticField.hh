//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/CylMapMagneticField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4FieldManager.hh>
#include <G4MagneticField.hh>
#include <G4TransportationManager.hh>
#include <corecel/Assert.hh>

#include "corecel/Types.hh"
#include "corecel/math/Quantity.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/CylMapField.hh"
#include "celeritas/field/CylMapFieldParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Generate field input with user-defined grid
CylMapFieldParams::Input
MakeCylMapFieldInput(std::vector<real_type> const& r_grid,
                     std::vector<real_type> const& phi_values,
                     std::vector<real_type> const& z_grid);

//---------------------------------------------------------------------------//
/*!
 * A user magnetic field equivalent to celeritas::CylMapField.
 */
class CylMapMagneticField : public G4MagneticField
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstFieldParams = std::shared_ptr<CylMapFieldParams const>;
    //!@}

  public:
    // Construct with CylMapFieldParams
    inline explicit CylMapMagneticField(SPConstFieldParams field_params);

    // Calculate values of the magnetic field vector
    inline void
    GetFieldValue(double const point[3], double* field) const override;

  private:
    SPConstFieldParams params_;
    CylMapField calc_field_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with the Celeritas shared CylMapFieldParams.
 */
CylMapMagneticField::CylMapMagneticField(SPConstFieldParams params)
    : params_(std::move(params))
    , calc_field_(CylMapField{params_->ref<MemSpace::native>()})
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the magnetic field vector at the given position.
 */
void CylMapMagneticField::GetFieldValue(double const pos[3], double* field) const
{
    // Calculate the magnetic field value in the native Celeritas unit system
    Real3 result = calc_field_(convert_from_geant(pos, clhep_length));
    for (auto i = 0; i < 3; ++i)
    {
        // Return values of the field vector in CLHEP::tesla for Geant4
        auto ft = native_value_to<units::FieldTesla>(result[i]);
        field[i] = convert_to_geant(ft.value(), CLHEP::tesla);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

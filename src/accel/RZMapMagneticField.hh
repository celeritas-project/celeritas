//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/RZMapMagneticField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4MagneticField.hh>

#include "corecel/Macros.hh"
#include "corecel/math/ArrayOperators.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/ext/Convert.geant.hh"
#include "celeritas/field/RZMapField.hh"
#include "celeritas/field/RZMapFieldParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A user magnetic field equivalent to celeritas::RZMapField.
 */
class RZMapMagneticField : public G4MagneticField
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstFieldParams = std::shared_ptr<RZMapFieldParams const>;
    //!@}

  public:
    // Construct with RZMapFieldParams
    inline explicit RZMapMagneticField(SPConstFieldParams field_params);

    // Calculate values of the magnetic field vector
    inline void GetFieldValue(double const point[3], double* field) const;

  private:
    SPConstFieldParams params_;
    RZMapField calc_field_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with the Celeritas shared RZMapFieldParams.
 */
RZMapMagneticField::RZMapMagneticField(SPConstFieldParams params)
    : params_(std::move(params))
    , calc_field_(RZMapField{params_->ref<MemSpace::native>()})
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the magnetic field vector at the given position.
 */
void RZMapMagneticField::GetFieldValue(double const pos[3], double* field) const
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

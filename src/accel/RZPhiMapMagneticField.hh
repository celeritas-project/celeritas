//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/RZMapMagneticField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4FieldManager.hh>
#include <G4MagneticField.hh>
#include <G4TransportationManager.hh>

#include "corecel/Macros.hh"
#include "corecel/math/ArrayOperators.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/ext/GeantUnits.hh"
#include "celeritas/field/RZPhiMapField.hh"
#include "celeritas/field/RZPhiMapFieldParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A user magnetic field equivalent to celeritas::RZPhiMapField.
 */
class RZPhiMapMagneticField : public G4MagneticField
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstFieldParams = std::shared_ptr<RZPhiMapFieldParams const>;
    //!@}

  public:
    // Construct with RZPhiMapFieldParams
    inline explicit RZPhiMapMagneticField(SPConstFieldParams field_params);

    // Calculate values of the magnetic field vector
    inline void
    GetFieldValue(double const point[3], double* field) const override;

  private:
    SPConstFieldParams params_;
    RZPhiMapField calc_field_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with the Celeritas shared RZPhiMapFieldParams.
 */
RZPhiMapMagneticField::RZPhiMapMagneticField(SPConstFieldParams params)
    : params_(std::move(params))
    , calc_field_(RZPhiMapField{params_->ref<MemSpace::native>()})
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the magnetic field vector at the given position.
 */
void RZPhiMapMagneticField::GetFieldValue(double const pos[3],
                                          double* field) const
{
    auto const& g4_field = *G4TransportationManager::GetTransportationManager()
                                ->GetFieldManager()
                                ->GetDetectorField();
    double gfield[3];
    g4_field.GetFieldValue(pos, gfield);
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

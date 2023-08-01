//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MagneticField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4MagneticField.hh>
#include <globals.hh>

#include "corecel/Macros.hh"
#include "celeritas/field/RZMapField.hh"
#include "celeritas/field/RZMapFieldParams.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * A user magnetic field equivalent to celeritas::RZMapField.
 */
class MagneticField : public G4MagneticField
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstFieldParams = std::shared_ptr<RZMapFieldParams const>;
    //!@}

  public:
    // Construct with RZMapFieldParams
    inline explicit MagneticField(SPConstFieldParams field_params);

    // Calculate values of the magnetic field vector
    inline void GetFieldValue(double const point[3], double* field) const;

    //// COMMON PROPERTIES ////
    static constexpr double scale() { return CLHEP::tesla / units::tesla; }

  private:
    SPConstFieldParams params_;
    RZMapField calc_field_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with the Celeritas shared RZMapFieldParams.
 */
MagneticField::MagneticField(SPConstFieldParams params)
    : params_(std::move(params))
    , calc_field_(RZMapField{params_->ref<MemSpace::native>()})
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate values of the magnetic field vector at the given position
 * using the volume-based celetias::RZMapField.
 */
void MagneticField::GetFieldValue(double const pos[3], double* field) const
{
    // Calculate the magnetic field value in the native Celeritas unit system
    Real3 result = this->calc_field_(Real3{pos[0], pos[1], pos[2]});
    for (auto i = 0; i < 3; ++i)
    {
        // Return values of the field vector in CLHEP::tesla for Geant4
        field[i] = result[i] * this->scale();
    }
    printf("B[2]= %g\n", field[2]);
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

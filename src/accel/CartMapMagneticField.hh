//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/CartMapMagneticField.hh
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
#include "celeritas/field/CartMapField.hh"
#include "celeritas/field/CartMapFieldParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! POD struct for CartMap field grid parameters
struct CartMapFieldGridParams
{
    G4double min_x;  //!< Minimum X coordinate
    G4double max_x;  //!< Maximum X coordinate
    size_type num_x;  //!< Number of grid points in X direction

    G4double min_y;  //!< Minimum Y coordinate
    G4double max_y;  //!< Maximum Y coordinate
    size_type num_y;  //!< Number of grid points in Y direction

    G4double min_z;  //!< Minimum Z coordinate
    G4double max_z;  //!< Maximum Z coordinate
    size_type num_z;  //!< Number of grid points in Z direction
};

//---------------------------------------------------------------------------//
// Generate field input with user-defined uniform grid
CartMapFieldParams::Input
MakeCartMapFieldInput(CartMapFieldGridParams const& params);

//---------------------------------------------------------------------------//
/*!
 * A user magnetic field equivalent to celeritas::CartMapField.
 */
class CartMapMagneticField : public G4MagneticField
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstFieldParams = std::shared_ptr<CartMapFieldParams const>;
    //!@}

  public:
    // Construct with CartMapFieldParams
    inline explicit CartMapMagneticField(SPConstFieldParams field_params);

    // Calculate values of the magnetic field vector
    inline void
    GetFieldValue(G4double const point[3], G4double* field) const override;

  private:
    SPConstFieldParams params_;
    CartMapField calc_field_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with the Celeritas shared CartMapFieldParams.
 */
CartMapMagneticField::CartMapMagneticField(SPConstFieldParams params)
    : params_(std::move(params))
    , calc_field_(CartMapField{params_->ref<MemSpace::native>()})
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the magnetic field vector at the given position.
 */
void CartMapMagneticField::GetFieldValue(G4double const pos[3],
                                         G4double* field) const
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

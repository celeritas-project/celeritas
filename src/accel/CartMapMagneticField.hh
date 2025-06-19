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
    explicit CartMapMagneticField(SPConstFieldParams field_params);

    // Destructor
    ~CartMapMagneticField() override;

    // Calculate values of the magnetic field vector
    void GetFieldValue(G4double const point[3], G4double* field) const override;

  private:
    // Forward declaration for PIMPL
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
}  // namespace celeritas

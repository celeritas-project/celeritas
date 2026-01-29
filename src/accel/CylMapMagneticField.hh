//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/CylMapMagneticField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4FieldManager.hh>
#include <G4TransportationManager.hh>

#include "celeritas/field/CylMapField.hh"
#include "celeritas/field/CylMapFieldParams.hh"
#include "celeritas/g4/MagneticField.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Generate field input with user-defined grid and explicit field
CylMapFieldParams::Input
MakeCylMapFieldInput(G4Field const& field,
                     std::vector<G4double> const& r_grid,
                     std::vector<G4double> const& phi_values,
                     std::vector<G4double> const& z_grid);

// Generate field input with user-defined grid from global field
CylMapFieldParams::Input
MakeCylMapFieldInput(std::vector<G4double> const& r_grid,
                     std::vector<G4double> const& phi_values,
                     std::vector<G4double> const& z_grid);

//---------------------------------------------------------------------------//
/*!
 * A user magnetic field equivalent to celeritas::CylMapField.
 */
using CylMapMagneticField
    = celeritas::MagneticField<CylMapFieldParams, CylMapField>;

//---------------------------------------------------------------------------//
}  // namespace celeritas

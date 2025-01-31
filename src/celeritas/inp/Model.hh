//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Model.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <variant>

class G4VPhysicalVolume;

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Set up geometry/material model.
 */
struct Model
{
    //! Path to GDML (or ORANGE override) file, or Geant4 world
    std::variant<std::string, G4VPhysicalVolume const*> geometry;

    // TODO: Materials
    // TODO: Regions
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas

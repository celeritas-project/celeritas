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
 *
 * The geometry filename should almost always be a GDML path. As a temporary
 * measure we also support loading from a \c .org.json file if the \c
 * StandaloneInput::physics_import is a ROOT file with serialized physics data.
 */
struct Model
{
    //! Path to GDML file, or Geant4 world
    std::variant<std::string, G4VPhysicalVolume const*> geometry;

    // TODO: Materials
    // TODO: Regions
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas

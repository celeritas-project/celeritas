//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/FromGeant.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "celeritas/inp/Model.hh"

namespace celeritas
{
namespace setup
{
//---------------------------------------------------------------------------//
// Load surfaces from global data
inp::Surfaces surfaces_from_geant();

//---------------------------------------------------------------------------//
// Load a model from a Geant4 world
inp::Model model_from_geant(G4VPhysicalVolume const*);

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4
inline inp::Model model_from_geant(G4VPhysicalVolume const*)
{
    CELER_NOT_CONFIGURED("Geant4");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantGeometryImport.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"

class G4VPhysicalVolume;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Build a VecGeom geometry in-memory from Geant4 (hiding Geant4 includes)
void g4_to_vecgeom(G4VPhysicalVolume const* world, bool verbose);

#if !CELERITAS_USE_GEANT4
inline void g4_to_vecgeom(G4VPhysicalVolume const*, bool)
{
    CELER_NOT_CONFIGURED("Geant4");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

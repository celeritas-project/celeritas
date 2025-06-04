//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/FromGeant.cc
//---------------------------------------------------------------------------//
#include "FromGeant.hh"

namespace celeritas
{
namespace setup
{
//---------------------------------------------------------------------------//
inp::Surfaces surfaces_from_geant()
{
    CELER_NOT_IMPLEMENTED("surfaces");
}

//---------------------------------------------------------------------------//
inp::Model model_from_geant(G4VPhysicalVolume const* world)
{
    CELER_VALIDATE(world, << "no world provided");
    inp::Model result;

    result.geometry = world;
    result.surfaces = surfaces_from_geant();

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas

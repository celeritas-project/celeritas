//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/EmExtraPhysicsHelper.cc
//---------------------------------------------------------------------------//
#include "EmExtraPhysicsHelper.hh"

#include <G4GammaNuclearXS.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Geant4 provided cross section classes
 */
EmExtraPhysicsHelper::EmExtraPhysicsHelper()
{
    gn_xs_ = std::make_shared<G4GammaNuclearXS>();
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the gamma-nuclear element cross section using G4GammaNuclearXS
 * in the native Geant4 unit [mb]
 */
double EmExtraPhysicsHelper::GammaNuclearElementXS(double energy, int z)
{
    return gn_xs_->ElementCrossSection(energy, z);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

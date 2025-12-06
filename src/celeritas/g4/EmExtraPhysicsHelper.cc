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
 * Calculate the gamma-nuclear element cross section using G4GammaNuclearXS.
 */
auto EmExtraPhysicsHelper::calc_gamma_nuclear_xs(AtomicNumber z,
                                                 MevEnergy energy) const
    -> MmSqXs
{
    MmSqXs xs;
    xs.value() = gn_xs_->ElementCrossSection(energy.value(), z.get());
    return xs;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

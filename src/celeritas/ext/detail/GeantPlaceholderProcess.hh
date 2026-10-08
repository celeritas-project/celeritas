//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantPlaceholderProcess.hh
//! \sa test/celeritas/ext/GeantTrackReconstruction.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <G4VProcess.hh>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Inert process reported as the creator of tracks created by Celeritas.
 *
 * Celeritas does not map its interactions to Geant4 processes, but user code
 * often dereferences \c G4Track::GetCreatorProcess for secondaries. This
 * process is never attached to a particle: all its physics methods are
 * no-ops. Its type is \c fUserDefined so that code classifying secondaries by
 * creator process type does not mistake it for a physics process.
 *
 * \warning Like any Geant4 process, an instance registers itself with the
 * thread-local process table, which deletes it when Geant4 clears its
 * thread-local state: allocate it with \c new on the thread that uses it, and
 * never delete it.
 */
class GeantPlaceholderProcess final : public G4VProcess
{
  public:
    // Construct with the "celeritas" name
    GeantPlaceholderProcess();

    //!@{
    //! \name Inert process interface
    G4double AlongStepGetPhysicalInteractionLength(
        G4Track const&, G4double, G4double, G4double&, G4GPILSelection*) final;
    G4double AtRestGetPhysicalInteractionLength(G4Track const&,
                                                G4ForceCondition*) final;
    G4double PostStepGetPhysicalInteractionLength(
        G4Track const&, G4double, G4ForceCondition*) final;
    G4VParticleChange* AlongStepDoIt(G4Track const&, G4Step const&) final;
    G4VParticleChange* AtRestDoIt(G4Track const&, G4Step const&) final;
    G4VParticleChange* PostStepDoIt(G4Track const&, G4Step const&) final;
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

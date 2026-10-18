//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantPlaceholderProcess.cc
//---------------------------------------------------------------------------//
#include "GeantPlaceholderProcess.hh"

#include <limits>
#include <G4ProcessType.hh>
#include <G4Track.hh>

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
constexpr G4double never = std::numeric_limits<G4double>::max();

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with the "celeritas" name.
 */
GeantPlaceholderProcess::GeantPlaceholderProcess()
    : G4VProcess("celeritas", fUserDefined)
{
    pParticleChange = &aParticleChange;
}

//---------------------------------------------------------------------------//
//! Never limit the step
G4double GeantPlaceholderProcess::AlongStepGetPhysicalInteractionLength(
    G4Track const&, G4double, G4double, G4double&, G4GPILSelection* selection)
{
    if (selection)
    {
        *selection = NotCandidateForSelection;
    }
    return never;
}

//---------------------------------------------------------------------------//
//! Never occur at rest
G4double GeantPlaceholderProcess::AtRestGetPhysicalInteractionLength(
    G4Track const&, G4ForceCondition* condition)
{
    if (condition)
    {
        *condition = NotForced;
    }
    return never;
}

//---------------------------------------------------------------------------//
//! Never occur after a step
G4double GeantPlaceholderProcess::PostStepGetPhysicalInteractionLength(
    G4Track const&, G4double, G4ForceCondition* condition)
{
    if (condition)
    {
        *condition = NotForced;
    }
    return never;
}

//---------------------------------------------------------------------------//
//! Leave the track unchanged
G4VParticleChange* GeantPlaceholderProcess::AlongStepDoIt(G4Track const& track,
                                                          G4Step const&)
{
    pParticleChange->Initialize(track);
    return pParticleChange;
}

//---------------------------------------------------------------------------//
//! Leave the track unchanged
G4VParticleChange* GeantPlaceholderProcess::AtRestDoIt(G4Track const& track,
                                                       G4Step const&)
{
    pParticleChange->Initialize(track);
    return pParticleChange;
}

//---------------------------------------------------------------------------//
//! Leave the track unchanged
G4VParticleChange* GeantPlaceholderProcess::PostStepDoIt(G4Track const& track,
                                                         G4Step const&)
{
    pParticleChange->Initialize(track);
    return pParticleChange;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

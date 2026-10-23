//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/NullGeantProcess.cc
//---------------------------------------------------------------------------//
#include "NullGeantProcess.hh"

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
//! Interaction length (or time at rest) of a process that never occurs
constexpr G4double no_interaction_length = std::numeric_limits<G4double>::max();

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with a process name.
 */
NullGeantProcess::NullGeantProcess(std::string const& name)
    : G4VProcess(name, fUserDefined)
{
    pParticleChange = &aParticleChange;
}

//---------------------------------------------------------------------------//
//! Never limit the step
G4double NullGeantProcess::AlongStepGetPhysicalInteractionLength(
    G4Track const&, G4double, G4double, G4double&, G4GPILSelection* selection)
{
    if (selection)
    {
        *selection = NotCandidateForSelection;
    }
    return no_interaction_length;
}

//---------------------------------------------------------------------------//
//! Never occur at rest
G4double NullGeantProcess::AtRestGetPhysicalInteractionLength(
    G4Track const&, G4ForceCondition* condition)
{
    if (condition)
    {
        *condition = NotForced;
    }
    return no_interaction_length;
}

//---------------------------------------------------------------------------//
//! Never occur after a step
G4double NullGeantProcess::PostStepGetPhysicalInteractionLength(
    G4Track const&, G4double, G4ForceCondition* condition)
{
    if (condition)
    {
        *condition = NotForced;
    }
    return no_interaction_length;
}

//---------------------------------------------------------------------------//
//! Leave the track unchanged
G4VParticleChange* NullGeantProcess::AlongStepDoIt(G4Track const& track,
                                                   G4Step const&)
{
    pParticleChange->Initialize(track);
    return pParticleChange;
}

//---------------------------------------------------------------------------//
//! Leave the track unchanged
G4VParticleChange* NullGeantProcess::AtRestDoIt(G4Track const& track,
                                                G4Step const&)
{
    pParticleChange->Initialize(track);
    return pParticleChange;
}

//---------------------------------------------------------------------------//
//! Leave the track unchanged
G4VParticleChange* NullGeantProcess::PostStepDoIt(G4Track const& track,
                                                  G4Step const&)
{
    pParticleChange->Initialize(track);
    return pParticleChange;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

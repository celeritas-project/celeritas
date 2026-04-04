//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/HadronicInteractor.cc
//---------------------------------------------------------------------------//
#include "HadronicInteractor.hh"

#include <G4HadronicProcess.hh>
#include <G4HadronicProcessStore.hh>
#include <G4ParticleDefinition.hh>
#include <G4Step.hh>
#include <G4Track.hh>
#include <G4VParticleChange.hh>

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct using a Geant4 particle definition and hadronic process type.
 */
HadronicInteractor::HadronicInteractor(G4ParticleDefinition const& particle,
                                       G4HadronicProcessType type)
    : particle_(particle)
{
    auto* proc_store = G4HadronicProcessStore::Instance();
    CELER_ASSERT(proc_store);

    process_ = proc_store->FindProcess(&particle_, type);

    CELER_VALIDATE(process_,
                   << "Hadronic process of type " << type
                   << " does not exist for particle "
                   << particle_.GetParticleName());
}

//---------------------------------------------------------------------------//
/*!
 * Invoke the PostStepDoIt action of the Geant4 hadronic process for the given
 * track and step.
 */
G4VParticleChange& HadronicInteractor::operator()(G4Track& track)
{
    CELER_EXPECT(track.GetParticleDefinition() == &particle_);

    process_->StartTracking(&track);
    auto* result = process_->PostStepDoIt(track, *track.GetStep());
    CELER_ASSERT(result);

    return *result;
}

//---------------------------------------------------------------------------//
/*!
 * Return the process name.
 */
std::string const& HadronicInteractor::process_name() const
{
    return process_->GetProcessName();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/HadronicInteractor.cc
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
    process_ = proc_store->FindProcess(&particle_, type);

    if (!process_)
    {
        G4ExceptionDescription description;
        description << "Hadronic process of type " << type
                    << " does not exist for particle "
                    << particle_.GetParticleName();

        G4Exception("HadronicInteractor::HadronicInteractor",
                    "ProcessNotFound",
                    FatalException,
                    description);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Invoke the PostStepDoIt action of the Geant4 hadronic process for the given
 * track and step.
 */
G4VParticleChange*
HadronicInteractor::PostStepDoIt(G4Track const& track, G4Step const& step)
{
    CELER_EXPECT(track.GetParticleDefinition() == &particle_);

    process_->StartTracking(const_cast<G4Track*>(&track));

    return process_->PostStepDoIt(track, step);
}

//---------------------------------------------------------------------------//
/*!
 * Return the process name.
 */
G4String HadronicInteractor::GetProcessName() const
{
    return process_->GetProcessName();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

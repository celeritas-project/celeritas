//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/offload-template/src/RunAction.cc
//---------------------------------------------------------------------------//
#include "RunAction.hh"

#include "Celeritas.hh"
#include "G4Threading.hh"

//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
RunAction::RunAction() : G4UserRunAction() {}

//---------------------------------------------------------------------------//
/*!
 * Initialize master and worker threads in Celeritas.
 */
void RunAction::BeginOfRunAction(G4Run const* run)
{
    CelerSimpleOffload().BeginOfRunAction(run);

    auto& shared_params = CelerSharedParams();
    // Add Celeritas tracking manager to electron, positron, gamma.
    CELER_ASSERT(shared_params);
    if (shared_params.StatusMode() != celeritas::SharedParams::Mode::disabled)
    {
        CELER_LOG_LOCAL(debug) << "Activating tracking manager";
        auto tm = std::make_unique<celeritas::TrackingManager>(
            &shared_params, &CelerLocalTransporter());

        for (G4ParticleDefinition* particle : shared_params.OffloadParticles())
        {
            particle->SetTrackingManager(tm.get());
        }

        // Intentionally leak tm since Geant4 doesn't support shared
        // pointer semantics
        tm.release();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clear local data and return Celeritas to an invalid state.
 */
void RunAction::EndOfRunAction(G4Run const* run)
{
    CelerSimpleOffload().EndOfRunAction(run);
}

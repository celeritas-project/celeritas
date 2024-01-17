//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManagerOffload.cc
//---------------------------------------------------------------------------//
#include "TrackingManagerOffload.hh"

#include <G4ProcessManager.hh>
#include <G4ProcessVector.hh>
#include <G4Track.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"

#include "ExceptionConverter.hh"
#include "LocalTransporter.hh"
#include "SharedParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct a tracking manager with data needed to offload to Celeritas.
 *
 * TODO: Clarify thread-locality. Construction/addition to \c
 * G4ParticleDefinition appears to take place on the master thread, typically
 * in the ConstructProcess method, but the tracking manager pointer is part of
 * the split-class data for the particle. It's observed that different threads
 * have distinct pointers to a LocalTransporter instance, and that these match
 * those of the global thread-local instances in test problems.
 */
TrackingManagerOffload::TrackingManagerOffload(SharedParams const* params,
                                               LocalTransporter* local)
    : params_(params), transport_(local)
{
    CELER_EXPECT(params);
    CELER_EXPECT(local);
}

//---------------------------------------------------------------------------//
/*!
 * Build physics tables for this particle.
 *
 * Messaged by the \c G4ParticleDefinition who stores us whenever cross-section
 * tables have to be rebuilt (i.e. if new materials have been defined). An
 * override is needed for Celeritas as it uses the particle's process manager
 * and tables to initialize its own physics data for the particle, and this is
 * disabled when a custom tracking manager is used. Note that this also means
 * we could have filters in HandOverOneTrack to hand back the track to the
 * general G4TrackingManager if matching a predicate(s).
 *
 * The implementation follows that in \c G4VUserPhysicsList::BuildPhysicsTable
 * , see also Geant4 Extended Example runAndEvent/RE07.
 */
void TrackingManagerOffload::BuildPhysicsTable(G4ParticleDefinition const& part)
{
    G4ProcessManager* pManagerShadow = part.GetMasterProcessManager();
    G4ProcessManager* pManager = part.GetProcessManager();
    CELER_ASSERT(pManager);

    G4ProcessVector* pVector = pManager->GetProcessList();
    CELER_ASSERT(pVector);
    for (auto j : range(pVector->size()))
    {
        G4VProcess* proc = (*pVector)[j];
        if (pManagerShadow == pManager)
        {
            proc->BuildPhysicsTable(part);
        }
        else
        {
            proc->BuildWorkerPhysicsTable(part);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Prepare physics tables for this particle.
 *
 * Messaged by the \c G4ParticleDefinition who stores us whenever cross-section
 * tables have to be rebuilt (i.e. if new materials have been defined). As with
 * BuildPhysicsTable, we override this to ensure all Geant4
 * process/cross-section data is available for Celeritas to use.
 *
 * The implementation follows that in \c
 * G4VUserPhysicsList::PreparePhysicsTable , see also Geant4 Extended Example
 * runAndEvent/RE07.
 */
void TrackingManagerOffload::PreparePhysicsTable(G4ParticleDefinition const& part)
{
    G4ProcessManager* pManagerShadow = part.GetMasterProcessManager();
    G4ProcessManager* pManager = part.GetProcessManager();
    CELER_ASSERT(pManager);

    G4ProcessVector* pVector = pManager->GetProcessList();
    CELER_ASSERT(pVector);
    for (auto j : range(pVector->size()))
    {
        G4VProcess* proc = (*pVector)[j];
        if (pManagerShadow == pManager)
        {
            proc->PreparePhysicsTable(part);
        }
        else
        {
            proc->PrepareWorkerPhysicsTable(part);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Offload the incoming track to Celeritas.
 */
void TrackingManagerOffload::HandOverOneTrack(G4Track* track)
{
    CELER_EXPECT(track);
    CELER_EXPECT(*transport_);

    // Offload this track to Celeritas for transport
    ExceptionConverter call_g4exception{"celer0001", params_};
    CELER_TRY_HANDLE(transport_->Push(*track), call_g4exception);

    // G4VTrackingManager takes ownership, so kill Geant4 track
    track->SetTrackStatus(fStopAndKill);
    delete track;
}

//---------------------------------------------------------------------------//
/*!
 * Complete processing of any buffered tracks.
 *
 * Note that this is called in \c G4EventManager::DoProcessing(G4Event*) after
 * the after the main tracking loop has completed.
 *
 * That is done to allow for models that may add "onload" particles back to
 * Geant4.
 */
void TrackingManagerOffload::FlushEvent()
{
    ExceptionConverter call_g4exception{"celer0002", params_};
    CELER_TRY_HANDLE(transport_->Flush(), call_g4exception);
}

}  // namespace celeritas

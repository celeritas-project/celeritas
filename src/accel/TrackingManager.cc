//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManager.cc
//---------------------------------------------------------------------------//
#include "TrackingManager.hh"

#include <G4EventManager.hh>
#include <G4ProcessManager.hh>
#include <G4ProcessVector.hh>
#include <G4Track.hh>
#include <G4TrackingManager.hh>
#include <G4UserTrackingAction.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"

#include "ExceptionConverter.hh"
#include "SharedParams.hh"
#include "TrackOffloadInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct a tracking manager with data needed to offload to Celeritas.
 *
 * \note The shared/local pointers must remain valid for the lifetime of the
 * run. The local transporter should be null on the "master" thread of an MT
 * run.
 */
TrackingManager::TrackingManager(SharedParams const* params,
                                 TrackOffloadInterface* local)
    : params_(params), transport_(local)
{
    CELER_EXPECT(params_);
    CELER_EXPECT(static_cast<bool>(transport_)
                 == !(G4Threading::IsMasterThread()
                      && G4Threading::IsMultithreadedApplication()));
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
void TrackingManager::BuildPhysicsTable(G4ParticleDefinition const& part)
{
    CELER_EXPECT(params_->mode() != SharedParams::Mode::disabled);

    CELER_LOG_LOCAL(debug) << "Building physics table for "
                           << part.GetParticleName();

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
 * \c BuildPhysicsTable, we override this to ensure all Geant4
 * process/cross-section data is available for Celeritas to use.
 *
 * The implementation follows that in \c
 * G4VUserPhysicsList::PreparePhysicsTable , see also Geant4 Extended Example
 * runAndEvent/RE07.
 */
void TrackingManager::PreparePhysicsTable(G4ParticleDefinition const& part)
{
    CELER_EXPECT(params_->mode() != SharedParams::Mode::disabled);

    CELER_LOG_LOCAL(debug) << "Preparing physics table for "
                           << part.GetParticleName();

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
 *
 * This will \em not be called in the "master" thread of an MT run.
 *
 * Because the custom tracking manager completely bypasses Geant4's standard
 * \c G4TrackingManager::ProcessOneTrack , the \c G4UserTrackingAction
 * callbacks are never fired. Frameworks such as DD4hep register MC-truth
 * bookkeeping (e.g.\ \c Geant4ParticleHandler ) on those callbacks, and
 * missing them leads to an inconsistent particle record and crashes at
 * end-of-event. We therefore manually invoke the pre- and post-tracking
 * user actions around the offload so that every intercepted track is still
 * visible to the rest of the framework.
 */
void TrackingManager::HandOverOneTrack(G4Track* track)
{
    CELER_EXPECT(track);
    CELER_EXPECT(transport_);

    if (CELER_UNLIKELY(!validated_))
    {
        CELER_TRY_HANDLE(
            CELER_VALIDATE(
                params_->mode()
                    == (*transport_ ? SharedParams::Mode::enabled
                                    : SharedParams::Mode::kill_offload),
                << "Celeritas was not initialized properly (maybe "
                   "BeginOfRunAction was not called?)"),
            ExceptionConverter("celer.track.validate"));
        validated_ = true;
    }

    // Notify user tracking actions (e.g. DD4hep's ParticleHandler) so they
    // can maintain MC truth bookkeeping for this track even though it will
    // not be stepped by the standard G4TrackingManager.
    G4UserTrackingAction* user_action = G4EventManager::GetEventManager()
                                            ->GetTrackingManager()
                                            ->GetUserTrackingAction();

    if (user_action)
    {
        user_action->PreUserTrackingAction(track);
    }

    if (*transport_)
    {
        // Offload this track to Celeritas for transport
        CELER_TRY_HANDLE(transport_->Push(*track),
                         ExceptionConverter("celer.track.push", params_));
    }

    // Mark track as killed before firing the post-action so that framework
    // bookkeeping sees the correct final status.
    track->SetTrackStatus(fStopAndKill);

    if (user_action)
    {
        user_action->PostUserTrackingAction(track);
    }

    // G4VTrackingManager owns the track; delete it now
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
void TrackingManager::FlushEvent()
{
    if (*transport_)
    {
        CELER_TRY_HANDLE(transport_->Flush(),
                         ExceptionConverter("celer.event.flush", params_));
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

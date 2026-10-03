//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantTrackHandBack.cc
//---------------------------------------------------------------------------//
#include "GeantTrackHandBack.hh"

#include <utility>
#include <G4Event.hh>
#include <G4EventManager.hh>
#include <G4StackManager.hh>
#include <G4Track.hh>
#include <G4TrackingManager.hh>
#include <G4TrajectoryContainer.hh>
#include <G4VTrajectory.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
G4EventManager& event_manager()
{
    auto* result = G4EventManager::GetEventManager();
    CELER_ASSERT(result);
    return *result;
}

//---------------------------------------------------------------------------//
/*!
 * Store a trajectory in the current event, like G4EventManager.
 */
void store_trajectory(G4VTrajectory* trajectory)
{
    if (!trajectory)
    {
        return;
    }
    G4Event* event = event_manager().GetNonconstCurrentEvent();
    CELER_ASSERT(event);
    G4TrajectoryContainer* container = event->GetTrajectoryContainer();
    if (!container)
    {
        container = new G4TrajectoryContainer;
        event->SetTrajectoryContainer(container);
    }
    container->insert(trajectory);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with the thread-local track reconstruction.
 */
GeantTrackHandBack::GeantTrackHandBack(SPTrackReconstruction recon)
    : recon_{std::move(recon)}
{
    CELER_EXPECT(recon_);
}

//---------------------------------------------------------------------------//
/*!
 * Warn about tracks still pending in the Geant4 stack.
 */
GeantTrackHandBack::~GeantTrackHandBack()
{
    if (!handed_back_.empty())
    {
        CELER_LOG_LOCAL(warning)
            << handed_back_.size()
            << " handed-back tracks were not tracked by Geant4";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Return ownership of a reconstructed track to the current Geant4 event.
 *
 * The user stacking action may kill the track, in which case Geant4 deletes
 * it immediately along with any lent user information.
 */
void GeantTrackHandBack::operator()(UPTrack track, TrackOrigin origin)
{
    CELER_EXPECT(track);
    CELER_EXPECT((origin == TrackOrigin::secondary)
                 == (track->GetTrackID() == 0));

    // Ownership is transferred to Geant4
    G4Track* raw = track.release();
    handed_back_.insert(raw);
    ++num_handed_back_;
    if (!this->stack(raw, origin == TrackOrigin::offloaded))
    {
        // Track (and possibly lent user info) were deleted by Geant4
        handed_back_.erase(raw);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Track a handed-back track on CPU, returning false if not handed back.
 *
 * This takes ownership of the track if and only if it was handed back.
 */
bool GeantTrackHandBack::process(G4Track* track)
{
    CELER_EXPECT(track);
    if (handed_back_.erase(track) == 0)
    {
        return false;
    }

    auto& em = event_manager();
    G4TrackingManager* tm = em.GetTrackingManager();
    CELER_ASSERT(tm);

    tm->ProcessOneTrack(track);
    G4TrackStatus const status = track->GetTrackStatus();
    G4VTrajectory* trajectory = tm->GimmeTrajectory();
    G4TrackVector* secondaries = tm->GimmeSecondaries();

    switch (status)
    {
        case fStopButAlive:
        case fSuspend:
        case fSuspendAndWait:
        case fPostponeToNextEvent:
            // Stack the track again *without* the hand-back mark so that it
            // is offloaded to Celeritas when popped
            store_trajectory(trajectory);
            static_cast<void>(this->stack(track, /* id_already_set = */ true));
            em.StackTracks(secondaries);
            break;
        case fStopAndKill:
            store_trajectory(trajectory);
            em.StackTracks(secondaries);
            recon_->release(*track);
            delete track;
            break;
        case fKillTrackAndSecondaries:
            store_trajectory(trajectory);
            if (secondaries)
            {
                for (G4Track* sec : *secondaries)
                {
                    delete sec;
                }
                secondaries->clear();
            }
            recon_->release(*track);
            delete track;
            break;
        case fAlive:
            CELER_LOG_LOCAL(error)
                << "Illegal track status returned from G4TrackingManager "
                   "for handed-back track";
            store_trajectory(trajectory);
            em.StackTracks(secondaries);
            recon_->release(*track);
            delete track;
            break;
    }
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Push a track to the Geant4 stack, returning false if it was killed.
 *
 * A user stacking action that classifies the track as \c fKill causes the
 * stack manager to delete it. That is detected by the unchanged number of
 * stacked tracks, in which case the reconstruction forgets any user
 * information lent to the (now deleted) track.
 */
bool GeantTrackHandBack::stack(G4Track* track, bool id_already_set)
{
    auto& em = event_manager();
    G4StackManager* sm = em.GetStackManager();
    CELER_ASSERT(sm);

    auto const num_before = sm->GetNTotalTrack();
    G4TrackVector tracks{track};
    em.StackTracks(&tracks, id_already_set);
    if (sm->GetNTotalTrack() > num_before)
    {
        return true;
    }

    // The track was deleted, or moved to a stack that is not counted: in
    // either case Geant4 owns any user information now attached to it
    recon_->forfeit(track);
    return false;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

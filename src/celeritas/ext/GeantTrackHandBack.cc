//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantTrackHandBack.cc
//---------------------------------------------------------------------------//
#include "GeantTrackHandBack.hh"

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
 * it immediately.
 */
void GeantTrackHandBack::operator()(UPTrack track)
{
    CELER_EXPECT(track);
    CELER_EXPECT(track->GetTrackID() > 0);

    // Ownership is transferred to Geant4
    G4Track* raw = track.release();
    handed_back_.insert(raw);
    ++num_handed_back_;
    if (!GeantTrackHandBack::stack(raw))
    {
        // Don't keep the address of a deleted track
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
            static_cast<void>(GeantTrackHandBack::stack(track));
            em.StackTracks(secondaries);
            break;
        case fStopAndKill:
            store_trajectory(trajectory);
            em.StackTracks(secondaries);
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
            delete track;
            break;
        case fAlive:
            CELER_LOG_LOCAL(error)
                << "Illegal track status returned from G4TrackingManager "
                   "for handed-back track";
            store_trajectory(trajectory);
            em.StackTracks(secondaries);
            delete track;
            break;
    }
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Push a track with its existing ID to the Geant4 stack.
 *
 * A user stacking action that classifies the track as \c fKill causes the
 * stack manager to delete it. That is detected by the unchanged number of
 * stacked tracks, so that the address of the deleted track is not kept. (A
 * track moved to a sub-event stack, which is not counted, is treated the same
 * way: it will be offloaded again rather than tracked on CPU.)
 *
 * \return Whether the track was stacked
 */
bool GeantTrackHandBack::stack(G4Track* track)
{
    auto& em = event_manager();
    G4StackManager* sm = em.GetStackManager();
    CELER_ASSERT(sm);

    auto const num_before = sm->GetNTotalTrack();
    G4TrackVector tracks{track};
    em.StackTracks(&tracks, /* IDhasAlreadySet = */ true);
    return sm->GetNTotalTrack() > num_before;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

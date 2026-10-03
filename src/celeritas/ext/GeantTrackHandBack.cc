//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantTrackHandBack.cc
//---------------------------------------------------------------------------//
#include "GeantTrackHandBack.hh"

#include <algorithm>
#include <G4Event.hh>
#include <G4EventManager.hh>
#include <G4StackManager.hh>
#include <G4Track.hh>
#include <G4TrackingManager.hh>
#include <G4TrajectoryContainer.hh>
#include <G4VTrajectory.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

#include "detail/GeantTrackOrder.hh"

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
 * Warn about tracks not flushed or still pending in the Geant4 stack.
 */
GeantTrackHandBack::~GeantTrackHandBack()
{
    if (!deferred_.empty())
    {
        CELER_LOG_LOCAL(warning)
            << deferred_.size()
            << " handed-back tracks were never returned to Geant4";
    }
    if (!handed_back_.empty())
    {
        CELER_LOG_LOCAL(warning)
            << handed_back_.size()
            << " handed-back tracks were not tracked by Geant4";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Defer returning a reconstructed track to the current Geant4 event.
 */
void GeantTrackHandBack::operator()(UPTrack track)
{
    CELER_EXPECT(track);
    CELER_EXPECT(track->GetTrackID() > 0);

    deferred_.push_back(std::move(track));
}

//---------------------------------------------------------------------------//
/*!
 * Return deferred tracks to the current Geant4 event in a fixed order.
 *
 * Ownership of each track is transferred to Geant4. The user stacking action
 * may kill a track, in which case Geant4 deletes it immediately.
 */
void GeantTrackHandBack::flush()
{
    std::sort(deferred_.begin(),
              deferred_.end(),
              [](UPTrack const& lhs, UPTrack const& rhs) {
                  return detail::GeantTrackOrder{}(*lhs, *rhs);
              });

    for (auto& track : deferred_)
    {
        G4Track* raw = track.release();
        handed_back_.insert(raw);
        ++num_handed_back_;
        auto stacked = GeantTrackHandBack::stack(raw);
        if (stacked == Stacked::killed)
        {
            // Don't keep the address of a deleted track
            handed_back_.erase(raw);
        }
        else if (stacked == Stacked::other && !warned_not_urgent_)
        {
            CELER_LOG_LOCAL(warning)
                << "Handed-back track " << raw->GetTrackID()
                << " was not classified as urgent by the user stacking "
                   "action: it may not be tracked in the current event";
            warned_not_urgent_ = true;
        }
    }
    deferred_.clear();
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
 * \return Whether the track was killed, stacked as urgent, or stacked in a
 * waiting or postponed stack
 */
auto GeantTrackHandBack::stack(G4Track* track) -> Stacked
{
    auto& em = event_manager();
    G4StackManager* sm = em.GetStackManager();
    CELER_ASSERT(sm);

    auto const num_before = sm->GetNTotalTrack();
    auto const num_urgent_before = sm->GetNUrgentTrack();
    G4TrackVector tracks{track};
    em.StackTracks(&tracks, /* IDhasAlreadySet = */ true);
    if (sm->GetNTotalTrack() == num_before)
    {
        return Stacked::killed;
    }
    return sm->GetNUrgentTrack() > num_urgent_before ? Stacked::urgent
                                                     : Stacked::other;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

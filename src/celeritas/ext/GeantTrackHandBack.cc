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
#include <G4GeometryTolerance.hh>
#include <G4StackManager.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>
#include <G4StepStatus.hh>
#include <G4Track.hh>
#include <G4TrackingManager.hh>
#include <G4TrajectoryContainer.hh>
#include <G4VTrajectory.hh>
#include <G4Version.hh>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "geocel/g4/Convert.hh"

#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
#    include <VecGeom/base/Global.h>
#endif

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
/*!
 * Distance (in Geant4 units) to move a track off the boundary it stopped on.
 *
 * The point must be beyond the surface tolerance of both the Geant4 geometry
 * and the Celeritas geometry. VecGeom's tolerance is in native Celeritas
 * length units, which depend on \c CELERITAS_UNITS . The Geant4 backend uses
 * the Geant4 tolerance, and ORANGE only treats points exactly on a surface as
 * ambiguous.
 */
double calc_boundary_push()
{
    double tol = G4GeometryTolerance::GetInstance()->GetSurfaceTolerance();
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
    tol = std::max(tol,
                   native_to_geant<lengthunits::ClhepLength>(
                       static_cast<real_type>(vecgeom::kTolerance)));
#endif
    return 100 * tol;
}

//---------------------------------------------------------------------------//
/*!
 * Move a track that stopped on a geometry boundary into the next volume.
 *
 * A track stopped by Geant4 on a boundary is located by its position alone
 * when it is offloaded again. The Geant4 and VecGeom geometry backends can
 * then place it in the volume it is leaving, where it may fail to take a
 * step, and ORANGE refuses to initialize a track exactly on a surface. Moving
 * the track along its direction beyond the geometry tolerances (see \c
 * calc_boundary_push ) removes the ambiguity, except for nearly tangent
 * directions. In single precision the move is usually lost when the position
 * is converted.
 *
 * This must be called right after the track was tracked: the step it points
 * to is shared by all tracks.
 */
void move_off_boundary(G4Track& track)
{
    G4Step const* step = track.GetStep();
    if (!step || step->GetPostStepPoint()->GetStepStatus() != fGeomBoundary)
    {
        return;
    }
    track.SetPosition(track.GetPosition()
                      + calc_boundary_push() * track.GetMomentumDirection());
}

//---------------------------------------------------------------------------//
/*!
 * Push new secondaries to the stack of the current event with new track IDs.
 *
 * This is what G4EventManager does with the secondaries of a tracked track.
 * \c G4EventManager::StackTracks is private before Geant4 11.0, which is also
 * the oldest version supporting offload through a tracking manager.
 */
void stack_secondaries(G4TrackVector* secondaries)
{
#if G4VERSION_NUMBER >= 1100
    event_manager().StackTracks(secondaries);
#else
    CELER_DISCARD(secondaries);
    CELER_NOT_IMPLEMENTED("handing back tracks with Geant4 older than 11.0");
#endif
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
        // The track originates where it is handed back, like its vertex
        track->SetOriginTouchableHandle(track->GetTouchableHandle());

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
#if G4VERSION_NUMBER >= 1120
        case fSuspendAndWait:
#endif
        case fPostponeToNextEvent:
            // Stack the track again *without* the hand-back mark so that it
            // is offloaded to Celeritas when popped
            store_trajectory(trajectory);
            move_off_boundary(*track);
            static_cast<void>(GeantTrackHandBack::stack(track));
            stack_secondaries(secondaries);
            break;
        case fStopAndKill:
            store_trajectory(trajectory);
            stack_secondaries(secondaries);
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
            stack_secondaries(secondaries);
            delete track;
            break;
    }
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Push a track with its existing ID to the Geant4 stack.
 *
 * Like \c G4EventManager does when it stacks a suspended track again, this
 * pushes the track directly to the stack manager (which calls the user
 * stacking action): \c G4EventManager::StackTracks would also use up a track
 * ID and overwrite the origin touchable of the track.
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
    sm->PushOneTrack(track);
    if (sm->GetNTotalTrack() == num_before)
    {
        return Stacked::killed;
    }
    return sm->GetNUrgentTrack() > num_urgent_before ? Stacked::urgent
                                                     : Stacked::other;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

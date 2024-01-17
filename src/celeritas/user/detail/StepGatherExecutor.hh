//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepGatherExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Gather step data for transfer to user hits.
 */
template<StepPoint P>
struct StepGatherExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<StepParamsData> const params;
    NativeRef<StepStateData> const state;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Gather step data on device based on the user selection.
 */
template<StepPoint P>
CELER_FUNCTION void
StepGatherExecutor<P>::operator()(celeritas::CoreTrackView const& track)
{
    CELER_EXPECT(params && state);

#define SGL_SET_IF_SELECTED(ATTR, VALUE)                          \
    do                                                            \
    {                                                             \
        if (this->params.selection.ATTR)                          \
        {                                                         \
            this->state.data.ATTR[track.track_slot_id()] = VALUE; \
        }                                                         \
    } while (0)

    {
        auto const sim = track.make_sim_view();
        bool inactive = (sim.status() == TrackStatus::inactive);

        if (P == StepPoint::post)
        {
            // Always save track ID to clear output from inactive slots
            this->state.data.track_id[track.track_slot_id()]
                = inactive ? TrackId{} : sim.track_id();
        }

        if (inactive)
        {
            if (P == StepPoint::pre && !this->params.detector.empty())
            {
                // Clear detector ID for inactive threads
                this->state.data.detector[track.track_slot_id()] = {};
            }

            // No more data to be written
            return;
        }
    }

    if (!this->params.detector.empty())
    {
        // Apply detector filter at beginning of step (volume in which we're
        // stepping)
        if (P == StepPoint::pre)
        {
            auto const geo = track.make_geo_view();
            CELER_ASSERT(!geo.is_outside());
            VolumeId vol = geo.volume_id();
            CELER_ASSERT(vol);

            // Map volume ID to detector ID
            this->state.data.detector[track.track_slot_id()]
                = this->params.detector[vol];
        }

        if (!this->state.data.detector[track.track_slot_id()])
        {
            // We're not in a sensitive detector: don't save any further data
            return;
        }

        if (P == StepPoint::post && this->params.nonzero_energy_deposition)
        {
            // Filter out tracks that didn't deposit energy over the step
            auto const pstep = track.make_physics_step_view();
            if (pstep.energy_deposition() == zero_quantity())
            {
                // Clear detector ID and stop recording
                this->state.data.detector[track.track_slot_id()] = {};
                return;
            }
        }
    }

    {
        auto const sim = track.make_sim_view();

        SGL_SET_IF_SELECTED(points[P].time, sim.time());
        if (P == StepPoint::post)
        {
            SGL_SET_IF_SELECTED(event_id, sim.event_id());
            SGL_SET_IF_SELECTED(parent_id, sim.parent_id());
            SGL_SET_IF_SELECTED(track_step_count, sim.num_steps());

            SGL_SET_IF_SELECTED(action_id, sim.post_step_action());
            SGL_SET_IF_SELECTED(step_length, sim.step_length());
        }
    }

    {
        auto const geo = track.make_geo_view();

        SGL_SET_IF_SELECTED(points[P].pos, geo.pos());
        SGL_SET_IF_SELECTED(points[P].dir, geo.dir());
        SGL_SET_IF_SELECTED(points[P].volume_id,
                            geo.is_outside() ? VolumeId{} : geo.volume_id());
    }

    {
        auto const par = track.make_particle_view();

        if (P == StepPoint::post)
        {
            auto const pstep = track.make_physics_step_view();
            SGL_SET_IF_SELECTED(energy_deposition, pstep.energy_deposition());
            SGL_SET_IF_SELECTED(particle, par.particle_id());
        }
        SGL_SET_IF_SELECTED(points[P].energy, par.energy());
    }
#undef SGL_SET_IF_SELECTED
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

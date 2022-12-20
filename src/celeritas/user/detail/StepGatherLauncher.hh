//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepGatherLauncher.hh
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
struct StepGatherLauncher
{
    //!@{
    //! \name Type aliases
    using CoreRefNative       = CoreRef<MemSpace::native>;
    using StepParamsRefNative = NativeCRef<StepParamsData>;
    using StepStateRefNative  = NativeRef<StepStateData>;
    //!@}

    //// DATA ////

    CoreRefNative const&       core_data;
    StepParamsRefNative const& step_params;
    StepStateRefNative const&  step_state;

    //// METHODS ////

    CELER_FUNCTION void operator()(ThreadId thread) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Gather step data on device based on the user selection.
 */
template<StepPoint P>
CELER_FUNCTION void StepGatherLauncher<P>::operator()(ThreadId thread) const
{
    CELER_ASSERT(thread < this->core_data.states.size());

    const celeritas::CoreTrackView track(
        this->core_data.params, this->core_data.states, thread);

#define SGL_SET_IF_SELECTED(ATTR, VALUE)           \
    do                                             \
    {                                              \
        if (this->step_params.selection.ATTR)      \
        {                                          \
            this->step_state.ATTR[thread] = VALUE; \
        }                                          \
    } while (0)

    {
        const auto sim      = track.make_sim_view();
        bool inactive = (sim.status() == TrackStatus::inactive);

        if (P == StepPoint::post)
        {
            // Always save track ID to clear output from inactive slots
            this->step_state.track_id[thread] = inactive ? TrackId{}
                                                         : sim.track_id();
        }

        if (inactive)
        {
            if (P == StepPoint::pre && !this->step_params.detector.empty())
            {
                // Clear detector ID for inactive threads
                this->step_state.detector[thread] = {};
            }

            // No more data to be written
            return;
        }
    }

    if (!this->step_params.detector.empty())
    {
        // Apply detector filter at beginning of step (volume in which we're
        // stepping)
        if (P == StepPoint::pre)
        {
            const auto geo = track.make_geo_view();
            CELER_ASSERT(!geo.is_outside());
            VolumeId vol = geo.volume_id();
            CELER_ASSERT(vol);

            // Map volume ID to detector ID
            this->step_state.detector[thread] = this->step_params.detector[vol];
        }

        if (!this->step_state.detector[thread])
        {
            // We're not in a sensitive detector: don't save any further data
            return;
        }

        if (P == StepPoint::post && this->step_params.nonzero_energy_deposition)
        {
            // Filter out tracks that didn't deposit energy over the step
            const auto pstep = track.make_physics_step_view();
            if (pstep.energy_deposition() == zero_quantity())
            {
                // Clear detector ID and stop recording
                this->step_state.detector[thread] = {};
                return;
            }
        }
    }

    {
        const auto sim = track.make_sim_view();

        SGL_SET_IF_SELECTED(points[P].time, sim.time());
        if (P == StepPoint::post)
        {
            SGL_SET_IF_SELECTED(event_id, sim.event_id());
            SGL_SET_IF_SELECTED(track_step_count, sim.num_steps());

            const auto& limit = sim.step_limit();
            SGL_SET_IF_SELECTED(action_id, limit.action);
            SGL_SET_IF_SELECTED(step_length, limit.step);
        }
    }

    {
        const auto geo = track.make_geo_view();

        SGL_SET_IF_SELECTED(points[P].pos, geo.pos());
        SGL_SET_IF_SELECTED(points[P].dir, geo.dir());
        SGL_SET_IF_SELECTED(points[P].volume_id,
                            geo.is_outside() ? VolumeId{} : geo.volume_id());
    }

    {
        const auto par = track.make_particle_view();

        if (P == StepPoint::post)
        {
            const auto pstep = track.make_physics_step_view();
            SGL_SET_IF_SELECTED(energy_deposition, pstep.energy_deposition());
            SGL_SET_IF_SELECTED(particle, par.particle_id());
        }
        SGL_SET_IF_SELECTED(points[P].energy, par.energy());
    }
#undef SGL_SET_IF_SELECTED
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

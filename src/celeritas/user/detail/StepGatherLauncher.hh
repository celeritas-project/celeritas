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

    const NativeRef<StepPointStateData>& step_point = step_state.points[P];

    {
        auto sim      = track.make_sim_view();
        bool inactive = (sim.status() == TrackStatus::inactive);

        if (P == StepPoint::post)
        {
            // Always save track ID to clear output from inactive slots
            step_state.track[thread] = inactive ? TrackId{} : sim.track_id();
        }

        if (inactive)
        {
            // No more data to be written
            return;
        }

        if (step_params.selection.sim)
        {
            if (P == StepPoint::post)
            {
                const auto& limit                   = sim.step_limit();
                step_state.event[thread]            = sim.event_id();
                step_state.track_step_count[thread] = sim.num_steps();
                step_state.action[thread]           = limit.action;
                step_state.step_length[thread]      = limit.step;
            }
            step_point.time[thread] = sim.time();
        }
    }

    if (step_params.selection.geo)
    {
        auto geo = track.make_geo_view();

        step_point.pos[thread]    = geo.pos();
        step_point.dir[thread]    = geo.dir();
        step_point.volume[thread] = geo.is_outside() ? VolumeId{}
                                                     : geo.volume_id();
    }

    if (step_params.selection.phys)
    {
        auto par = track.make_particle_view();

        if (P == StepPoint::post)
        {
            auto pstep                  = track.make_physics_step_view();
            step_state.particle[thread] = par.particle_id();
            step_state.energy_deposition[thread] = pstep.energy_deposition();
        }
        step_point.energy[thread] = par.energy();
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

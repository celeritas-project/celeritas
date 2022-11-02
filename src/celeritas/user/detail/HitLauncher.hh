//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/HitLauncher.hh
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
 * Gather hit data.
 */
class HitLauncher
{
    //!@{
    //! \name Type aliases
    using CoreRefNative      = CoreRef<MemSpace::native>;
    using HitParamsRefNative = NativeCRef<HitParamsData>;
    using HitStateRefNative  = NativeRef<HitStateData>;
    //!@}

    //// DATA ////

    CoreRefNative const&      core_data;
    HitParamsRefNative const& hit_params;
    HitStateRefNative const&  hit_state;

    //// METHODS ////

    CELER_FUNCTION void operator()(ThreadId thread) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Gather hits on device based on the user selection.
 */
CELER_FUNCTION void HitLauncher::operator()(ThreadId thread) const
{
    CELER_ASSERT(thread < this->core_data.states.size());
    const celeritas::CoreTrackView track(
        this->core_data.params, this->core_data.states, thread);

    {
        // Always save track ID to clear output from inactive slots
        auto sim                = track.make_sim_view();
        bool inactive           = (sim.status() == TrackStatus::inactive);
        hit_state.track[thread] = inactive ? TrackId{} : sim.track_id();

        if (inactive)
        {
            // No more data to be written
            return;
        }

        if (hit_params.selection.sim)
        {
            hit_state.event[thread]     = sim.event_id();
            hit_state.num_steps[thread] = sim.num_steps();
            hit_state.time[thread]      = sim.time();
        }

        if (hit_params.selection.post_step)
        {
            const auto& limit             = sim.step_limit();
            hit_state.step_length[thread] = limit.step;
            hit_state.action              = limit.action;
        }
    }

    if (hit_params.selection.geo)
    {
        auto geo = track.make_geo_view();

        hit_state.pos[thread]    = geo.pos();
        hit_state.dir[thread]    = geo.dir();
        hit_state.volume[thread] = geo.is_outside() ? VolumeId{}
                                                    : geo.volume_id();
    }

    if (hit_params.selection.phys)
    {
        auto par = track.make_particle_view();

        hit_state.particle[thread]       = par.particle_id();
        hit_state.kinetic_energy[thread] = par.energy();

        auto pstep = track.make_physics_step_view();

        hit_state.energy_deposition[thread] = hit_params.is_post_step
                                                  ? pstep.energy_deposition()
                                                  : zero_quantity();
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

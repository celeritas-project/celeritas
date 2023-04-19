//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/ActionDiagnosticLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/Atomics.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Kernel launcher for tallying post-step actions by particle type.
 */
struct ActionDiagnosticLauncher
{
    //// DATA ////

    NativeCRef<CoreParamsData> const& params;
    NativeRef<CoreStateData> const& state;
    NativeRef<ActionDiagnosticStateData>& data;

    //// METHODS ////

    inline CELER_FUNCTION void operator()(ThreadId tid);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Tally post-step actions by particle type.
 */
CELER_FUNCTION void ActionDiagnosticLauncher::operator()(ThreadId tid)
{
    using BinId = ItemId<size_type>;

    CELER_EXPECT(params);
    CELER_EXPECT(state);
    CELER_EXPECT(data);
    CELER_EXPECT(tid < state.size());

    celeritas::CoreTrackView const track(params, state, tid);
    ParticleId pid = track.make_particle_view().particle_id();
    CELER_ASSERT(pid);
    ActionId aid = track.make_sim_view().step_limit().action;
    CELER_ASSERT(aid);
    size_type num_particles = track.make_physics_view().num_particles();

    BinId bin{aid.unchecked_get() * num_particles + pid.unchecked_get()};
    CELER_ASSERT(bin < data.counts.size());
    celeritas::atomic_add(&data.counts[bin], size_type(1));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

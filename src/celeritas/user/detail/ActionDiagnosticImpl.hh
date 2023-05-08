//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/ActionDiagnosticImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/Atomics.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Tally post-step actions by particle type.
 */
inline CELER_FUNCTION void
tally_action(CoreTrackView const& track,
             NativeCRef<ParticleTallyParamsData> const& params,
             NativeRef<ParticleTallyStateData> const& state)
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    using BinId = ItemId<size_type>;

    auto action = track.make_sim_view().step_limit().action;
    CELER_ASSERT(action);
    auto particle = track.make_particle_view().particle_id();
    CELER_ASSERT(particle);

    BinId bin{particle.unchecked_get() * params.num_bins
              + action.unchecked_get()};
    CELER_ASSERT(bin < state.counts.size());
    celeritas::atomic_add(&state.counts[bin], size_type(1));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

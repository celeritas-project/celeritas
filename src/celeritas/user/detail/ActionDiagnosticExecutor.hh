//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/ActionDiagnosticExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/Atomics.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "../ParticleTallyData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Tally tracks that are active, have errors, *or* are killed.
 */
struct ActionDiagnosticCondition
{
    CELER_FUNCTION bool operator()(SimTrackView const& sim) const
    {
        return sim.status() != TrackStatus::inactive;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Tally post-step actions by particle type.
 */
struct ActionDiagnosticExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<ParticleTallyParamsData> const params;
    NativeRef<ParticleTallyStateData> const state;
};

//---------------------------------------------------------------------------//
CELER_FUNCTION void
ActionDiagnosticExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    using BinId = ItemId<size_type>;

    auto action = track.sim().post_step_action();
    CELER_ASSERT(action);
    auto particle = track.particle().particle_id();
    CELER_ASSERT(particle);

    BinId bin{particle.unchecked_get() * params.num_bins
              + action.unchecked_get()};
    CELER_ASSERT(bin < state.counts.size());
    celeritas::atomic_add(&state.counts[bin], size_type(1));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

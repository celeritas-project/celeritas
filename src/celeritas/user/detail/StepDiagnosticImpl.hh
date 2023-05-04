//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepDiagnosticImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Atomics.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Collect distribution of steps per track for each particle type.
 */
inline CELER_FUNCTION void
tally_steps(CoreTrackView const& track,
            NativeCRef<ParticleTallyParamsData> const& params,
            NativeRef<ParticleTallyStateData> const& state)
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    using BinId = ItemId<size_type>;

    // Tally the number of steps if the track was killed
    auto sim = track.make_sim_view();
    if (sim.status() == TrackStatus::killed)
    {
        // TODO: Add an ndarray-type class?
        auto get = [&params, &state](size_type i, size_type j) -> size_type& {
            size_type index = i * params.num_bins + j;
            CELER_ENSURE(index < state.counts.size());
            return state.counts[BinId(index)];
        };

        size_type num_steps
            = celeritas::min(sim.num_steps(), params.num_bins - 1);
        auto particle = track.make_particle_view().particle_id();

        // Increment the bin corresponding to the given particle and step count
        auto& bin = get(particle.get(), num_steps);
        atomic_add(&bin, size_type{1});
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

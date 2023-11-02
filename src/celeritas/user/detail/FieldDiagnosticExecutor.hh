//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/FieldDiagnosticExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Atomics.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "../FieldDiagnosticData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
struct FieldDiagnosticExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<FieldDiagnosticParamsData> const params;
    NativeRef<FieldDiagnosticStateData> const state;
};

//---------------------------------------------------------------------------//
/*!
 * Collect distribution of field substeps at each step.
 */
CELER_FUNCTION void
FieldDiagnosticExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    using BinId = ItemId<size_type>;

    auto particle = track.make_particle_view();
    if (particle.particle_view().charge() == zero_quantity())
    {
        return;
    }

    UniformGrid grid(params.energy);
    auto log_energy = std::log(particle.energy().value());
    if (log_energy < grid.front() || log_energy >= grid.back())
    {
        return;
    }

    auto get = [this](size_type i, size_type j) -> size_type& {
        size_type index = i * params.num_substep_bins + j;
        CELER_ENSURE(index < state.counts.size());
        return state.counts[BinId(index)];
    };

    size_type num_substeps = celeritas::min(
        track.make_sim_view().num_substeps(), params.num_substep_bins - 1);

    // Tally the number of field substeps
    auto& bin = get(grid.find(log_energy), num_substeps);
    atomic_add(&bin, size_type{1});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

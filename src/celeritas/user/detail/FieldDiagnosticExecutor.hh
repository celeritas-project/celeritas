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
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Atomics.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "../ParticleTallyData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
struct FieldDiagnosticExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<ParticleTallyParamsData> const params;
    NativeRef<ParticleTallyStateData> const state;
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

    if (track.make_particle_view().particle_view().charge() == zero_quantity())
    {
        return;
    }

    // Tally the number of field substeps
    size_type num_iter = celeritas::min(track.make_sim_view().num_substeps(),
                                        params.num_bins - 1);
    atomic_add(&state.counts[BinId(num_iter)], size_type{1});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

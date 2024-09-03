//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/CerenkovGeneratorExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/optical/CerenkovGenerator.hh"
#include "celeritas/optical/OffloadData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Generate Cerenkov photons from optical distribution data.
 */
struct CerenkovGeneratorExecutor
{
    //// TYPES ////

    using ParamsPtr = CRefPtr<CoreParamsData, MemSpace::native>;
    using StatePtr = RefPtr<CoreStateData, MemSpace::native>;

    //// DATA ////

    StatePtr state;
    // TODO: get material from optical params?
    NativeCRef<celeritas::optical::MaterialParamsData> const material;
    NativeCRef<celeritas::optical::CerenkovData> const cerenkov;
    NativeRef<OffloadStateData> const offload_state;
    RefPtr<celeritas::optical::CoreStateData, MemSpace::native> optical_state;
    OffloadBufferSize size;
    celeritas::optical::CoreStateCounters counters;

    //// FUNCTIONS ////

    // Generate optical primaries
    inline CELER_FUNCTION void operator()(CoreTrackView const& track) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Generate Cerenkov photons from optical distribution data.
 */
CELER_FUNCTION void
CerenkovGeneratorExecutor::operator()(CoreTrackView const& track) const
{
    CELER_EXPECT(state);
    CELER_EXPECT(cerenkov);
    CELER_EXPECT(material);
    CELER_EXPECT(offload_state);
    CELER_EXPECT(optical_state);
    CELER_EXPECT(size.cerenkov <= offload_state.cerenkov.size());

    using DistId = ItemId<celeritas::optical::GeneratorDistributionData>;
    using PrimaryId = ItemId<celeritas::optical::Primary>;

    // Threads may generate primaries from more than one distribution
    size_type dist_per_thread = ceil_div(size.cerenkov, state->size());
    for (auto i : range(dist_per_thread))
    {
        size_type dist_idx = i * state->size() + track.thread_id().get();
        if (dist_idx >= size.cerenkov)
            continue;

        auto const& dist = offload_state.cerenkov[DistId(dist_idx)];
        CELER_ASSERT(dist);

        // Get the offset in the primary buffer to start generating photons
        CELER_ASSERT(dist_idx < offload_state.offsets.size());
        auto start = counters.num_primaries
                     + offload_state.offsets[ItemId<size_type>(dist_idx)];

        optical::MaterialView opt_mat{material, dist.material};
        auto rng = track.make_rng_engine();

        celeritas::optical::CerenkovGenerator generate(opt_mat, cerenkov, dist);
        for (auto pid : range(PrimaryId(start), PrimaryId(dist.num_photons)))
        {
            CELER_ASSERT(pid < optical_state->init.primaries.size());
            optical_state->init.primaries[pid] = generate(rng);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

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

#include "OpticalUtils.hh"

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
    //// DATA ////

    RefPtr<CoreStateData, MemSpace::native> state;
    NativeCRef<celeritas::optical::MaterialParamsData> const material;
    NativeCRef<celeritas::optical::CerenkovData> const cerenkov;
    NativeRef<OffloadStateData> const offload_state;
    RefPtr<celeritas::optical::CoreStateData, MemSpace::native> optical_state;
    OffloadBufferSize size;

    //// FUNCTIONS ////

    // Generate optical track initializers
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
    using InitId = ItemId<celeritas::optical::TrackInitializer>;

    // Get the cumulative sum of the number of photons in the distributions.
    // Each bin gives the range of thread IDs that will generate from the
    // corresponding distribution
    auto offsets = offload_state.offsets[ItemRange<size_type>(
        ItemId<size_type>(0), ItemId<size_type>(size.cerenkov))];

    // Get the total number of initializers to generate
    size_type total_work = offsets.back();

    // Calculate the number of initializers for the thread to generate
    size_type local_work
        = calc_local_work(track.thread_id(), state->size(), total_work);

    auto rng = track.make_rng_engine();

    for (auto i : range(local_work))
    {
        // Calculate the index in the primary buffer this thread will write to
        size_type primary_idx = i * state->size() + track.thread_id().get();
        CELER_ASSERT(primary_idx < optical_state->init.initializers.size());

        // Find the distribution this thread will generate from
        size_type dist_idx = find_distribution_index(offsets, primary_idx);
        CELER_ASSERT(dist_idx < size.cerenkov);
        auto const& dist = offload_state.cerenkov[DistId(dist_idx)];
        CELER_ASSERT(dist);

        // Generate one primary from the distribution
        optical::MaterialView opt_mat{material, dist.material};
        celeritas::optical::CerenkovGenerator generate(opt_mat, cerenkov, dist);
        optical_state->init.initializers[InitId(primary_idx)] = generate(rng);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

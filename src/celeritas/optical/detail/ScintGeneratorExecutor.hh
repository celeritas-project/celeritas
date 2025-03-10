//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/ScintGeneratorExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/optical/OffloadData.hh"
#include "celeritas/optical/ScintillationGenerator.hh"
#include "celeritas/track/CoreStateCounters.hh"

#include "OpticalUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Generate scintillation photons from optical distribution data.
 */
struct ScintGeneratorExecutor
{
    //// DATA ////

    RefPtr<CoreStateData, MemSpace::native> state;
    NativeCRef<celeritas::optical::ScintillationData> const scintillation;
    NativeRef<OffloadStateData> const offload_state;
    RefPtr<celeritas::optical::CoreStateData, MemSpace::native> optical_state;
    OffloadBufferSize size;
    CoreStateCounters counters;

    //// FUNCTIONS ////

    // Generate optical track initializers
    inline CELER_FUNCTION void operator()(CoreTrackView const& track) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Generate scintillation photons from optical distribution data.
 */
CELER_FUNCTION void
ScintGeneratorExecutor::operator()(CoreTrackView const& track) const
{
    CELER_EXPECT(state);
    CELER_EXPECT(scintillation);
    CELER_EXPECT(offload_state);
    CELER_EXPECT(optical_state);
    CELER_EXPECT(size.scintillation <= offload_state.scintillation.size());

    using DistId = ItemId<celeritas::optical::GeneratorDistributionData>;
    using InitId = ItemId<celeritas::optical::TrackInitializer>;

    // Get the cumulative sum of the number of photons in the distributions.
    // Each bin gives the range of thread IDs that will generate from the
    // corresponding distribution
    auto offsets = offload_state.offsets[ItemRange<size_type>(
        ItemId<size_type>(0), ItemId<size_type>(size.scintillation))];

    // Get the total number of initializers to generate
    size_type total_work = offsets.back();

    // Calculate the number of initializers for the thread to generate
    size_type local_work = LocalWorkCalculator<size_type>{
        total_work, state->size()}(track.thread_id().get());

    auto rng = track.rng();

    for (auto i : range(local_work))
    {
        // Calculate the index in the initializer buffer (minus the offset)
        size_type idx = i * state->size() + track.thread_id().get();

        // Find the distribution this thread will generate from
        size_type dist_idx = find_distribution_index(offsets, idx);
        CELER_ASSERT(dist_idx < size.scintillation);
        auto const& dist = offload_state.scintillation[DistId(dist_idx)];
        CELER_ASSERT(dist);

        // Generate one primary from the distribution
        celeritas::optical::ScintillationGenerator generate(scintillation,
                                                            dist);
        size_type init_idx = counters.num_initializers + idx;
        CELER_ASSERT(init_idx < optical_state->init.initializers.size());
        optical_state->init.initializers[InitId(init_idx)] = generate(rng);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

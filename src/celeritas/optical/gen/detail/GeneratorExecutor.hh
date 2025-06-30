//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/MaterialData.hh"
#include "celeritas/track/CoreStateCounters.hh"

#include "GeneratorTraits.hh"
#include "OpticalGenAlgorithms.hh"
#include "../OffloadData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Generate photons from optical distribution data.
 */
template<GeneratorType G>
struct GeneratorExecutor
{
    //// TYPES ////

    template<Ownership W, MemSpace M>
    using Data = typename GeneratorTraits<G>::template Data<W, M>;
    using Generator = typename GeneratorTraits<G>::Generator;

    //// DATA ////

    RefPtr<celeritas::optical::CoreStateData, MemSpace::native> state;
    NativeCRef<celeritas::optical::MaterialParamsData> const material;
    NativeCRef<Data> const shared;
    NativeRef<GeneratorStateData> const offload;
    size_type buffer_size;
    CoreStateCounters counters;

    //// FUNCTIONS ////

    // Generate optical track initializers
    inline CELER_FUNCTION void operator()(optical::CoreTrackView const&) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Generate photons from optical distribution data.
 */
template<GeneratorType G>
CELER_FUNCTION void
GeneratorExecutor<G>::operator()(optical::CoreTrackView const& track) const
{
    CELER_EXPECT(state);
    CELER_EXPECT(shared);
    CELER_EXPECT(material);
    CELER_EXPECT(offload);
    CELER_EXPECT(buffer_size <= offload.distributions.size());

    using DistId = ItemId<GeneratorDistributionData>;
    using InitId = ItemId<celeritas::optical::TrackInitializer>;

    // Get the cumulative sum of the number of photons in the distributions.
    // Each bin gives the range of thread IDs that will generate from the
    // corresponding distribution
    auto offsets = offload.offsets[ItemRange<size_type>(
        ItemId<size_type>(0), ItemId<size_type>(buffer_size))];

    // Get the total number of initializers to generate
    size_type total_work = offsets.back();

    // Calculate the number of initializers for the thread to generate
    size_type local_work = LocalWorkCalculator<size_type>{
        total_work, state->size()}(track.track_slot_id().get());

    auto rng = track.rng();

    for (auto i : range(local_work))
    {
        // Calculate the index in the initializer buffer (minus the offset)
        size_type idx = i * state->size() + track.track_slot_id().get();

        // Find the distribution this thread will generate from
        size_type dist_idx = find_distribution_index(offsets, idx);
        CELER_ASSERT(dist_idx < buffer_size);
        auto const& dist = offload.distributions[DistId(dist_idx)];
        CELER_ASSERT(dist);

        // Generate one primary from the distribution
        optical::MaterialView opt_mat{material, dist.material};
        Generator generate(opt_mat, shared, dist);
        size_type init_idx = counters.num_initializers + idx;
        CELER_ASSERT(init_idx < state->init.initializers.size());
        state->init.initializers[InitId(init_idx)] = generate(rng);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

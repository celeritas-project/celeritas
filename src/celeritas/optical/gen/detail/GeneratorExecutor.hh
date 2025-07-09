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
#include "celeritas/track/detail/Utils.hh"

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

    CRefPtr<celeritas::optical::CoreParamsData, MemSpace::native> params;
    RefPtr<celeritas::optical::CoreStateData, MemSpace::native> state;
    NativeCRef<celeritas::optical::MaterialParamsData> const material;
    NativeCRef<Data> const shared;
    NativeRef<GeneratorStateData> const offload;
    size_type buffer_size{};
    CoreStateCounters counters;

    //// FUNCTIONS ////

    // Generate optical photons
    inline CELER_FUNCTION void operator()(TrackSlotId tid) const;
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        return (*this)(TrackSlotId{tid.unchecked_get()});
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Generate photons from optical distribution data.
 */
template<GeneratorType G>
CELER_FUNCTION void GeneratorExecutor<G>::operator()(TrackSlotId tid) const
{
    CELER_EXPECT(state);
    CELER_EXPECT(shared);
    CELER_EXPECT(material);
    CELER_EXPECT(offload);

    using DistId = ItemId<GeneratorDistributionData>;

    celeritas::optical::CoreTrackView track(*params, *state, tid);

    // Find the index of the first distribution that has a nonzero number of
    // primaries left to generate
    auto all_offsets = offload.offsets[ItemRange<size_type>(
        ItemId<size_type>(0), ItemId<size_type>(buffer_size))];
    auto buffer_start = celeritas::upper_bound(
        all_offsets.begin(), all_offsets.end(), size_type(0));
    CELER_ASSERT(buffer_start != all_offsets.end());

    // Get the cumulative sum of the number of photons in the distributions.
    // The values are used to determine which threads will generate from the
    // corresponding distribution
    Span<size_type> offsets{buffer_start, all_offsets.end()};

    // Find the distribution this thread will generate from
    size_type dist_idx = find_distribution_index(offsets, tid.get());
    CELER_ASSERT(dist_idx < offload.distributions.size());
    auto const& dist = offload.distributions[DistId(dist_idx)];
    CELER_ASSERT(dist);

    // Create the view to the new track to be initialized
    celeritas::optical::CoreTrackView vacancy{
        *params, *state, [&] {
            // Get the vacancy from the back in case there are more vacancies
            // than photons left to generate
            TrackSlotId idx{
                index_before(counters.num_vacancies, ThreadId(tid.get()))};
            return state->init.vacancies[idx];
        }()};

    // Generate one primary from the distribution
    auto rng = track.rng();
    optical::MaterialView opt_mat{material, dist.material};
    vacancy = Generator(opt_mat, shared, dist)(rng);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

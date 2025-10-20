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
#include "celeritas/track/Utils.hh"

#include "GeneratorAlgorithms.hh"
#include "../OffloadData.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Generate photons from optical distribution data.
 */
struct GeneratorExecutor
{
    //// DATA ////

    CRefPtr<CoreParamsData, MemSpace::native> params;
    RefPtr<CoreStateData, MemSpace::native> state;
    NativeCRef<MaterialParamsData> const material;
    NativeCRef<CherenkovData> const cherenkov;
    NativeCRef<ScintillationData> const scintillation;
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
CELER_FUNCTION void GeneratorExecutor::operator()(TrackSlotId tid) const
{
    CELER_EXPECT(state);
    CELER_EXPECT(material);
    CELER_EXPECT(offload);

    using DistId = ItemId<GeneratorDistributionData>;

    CoreTrackView track(*params, *state, tid);

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
    size_type dist_idx = buffer_start - all_offsets.begin()
                         + find_distribution_index(offsets, tid.get());
    CELER_ASSERT(dist_idx < offload.distributions.size());
    auto const& dist = offload.distributions[DistId(dist_idx)];
    CELER_ASSERT(dist);

    // Create the view to the new track to be initialized
    CoreTrackView vacancy{*params, *state, [&] {
                              // Get the vacancy from the back in case there
                              // are more vacancies than photons to generate
                              TrackSlotId idx{index_before(
                                  counters.num_vacancies, ThreadId(tid.get()))};
                              return state->init.vacancies[idx];
                          }()};

    // Generate one track from the distribution
    auto rng = track.rng();
    MaterialView opt_mat{material, dist.material};
    if (dist.type == GeneratorType::cherenkov)
    {
        CELER_ASSERT(cherenkov);
        vacancy = CherenkovGenerator(opt_mat, cherenkov, dist)(rng);
    }
    else
    {
        CELER_ASSERT(scintillation);
        vacancy = ScintillationGenerator(opt_mat, scintillation, dist)(rng);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas

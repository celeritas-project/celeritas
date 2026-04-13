//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/WlsGeneratorExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/WavelengthShiftData.hh"
#include "celeritas/optical/gen/WavelengthShiftGenerator.hh"
#include "celeritas/track/CoreStateCounters.hh"
#include "celeritas/track/Utils.hh"

#include "GeneratorAlgorithms.hh"

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
 * Generate WLS photons from optical distribution data.
 */
struct WlsGeneratorExecutor
{
    //// DATA ////

    CRefPtr<CoreParamsData, MemSpace::native> params;
    RefPtr<CoreStateData, MemSpace::native> state;
    NativeCRef<WavelengthShiftData> wls;
    NativeCRef<WavelengthShiftData> wls2;
    NativeRef<WlsGeneratorStateData> data;
    size_type buffer_size{};

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
 * Generate WLS photons from optical distribution data.
 */
CELER_FUNCTION void WlsGeneratorExecutor::operator()(TrackSlotId tid) const
{
    CELER_EXPECT(state);
    CELER_EXPECT(data);

    using DistId = ItemId<WlsDistributionData>;

    auto* counters = state->init.counters.data().get();

    // Get the cumulative sum of the number of photons in the distributions.
    // The values are used to determine which threads will generate from the
    // corresponding distribution
    auto offsets = data.offsets[ItemRange<size_type>(
        ItemId<size_type>(0), ItemId<size_type>(buffer_size))];

    // Find the distribution this thread will generate from
    size_type dist_idx = find_distribution_index(offsets, tid.get());
    CELER_ASSERT(dist_idx < data.distributions.size());
    auto& dist = data.distributions[DistId(dist_idx)];
    CELER_ASSERT(dist);

    // Create the view to the new track to be initialized
    CoreTrackView vacancy{
        *params, *state, [&] {
            // Get the vacancy from the back in case there are more vacancies
            // than photons to generate
            TrackSlotId idx{
                index_before(counters->num_vacancies, ThreadId(tid.get()))};
            return state->init.vacancies[idx];
        }()};

    // Generate one track from the distribution
    auto rng = vacancy.rng();
    if (dist.type == GeneratorType::wls)
    {
        CELER_ASSERT(wls);
        vacancy = WavelengthShiftGenerator(wls, dist)(rng);
    }
    else
    {
        CELER_ASSERT(wls2);
        vacancy = WavelengthShiftGenerator(wls2, dist)(rng);
    }

    // Update the number of photons left to generate
    atomic_add(&dist.num_photons, size_type(-1));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas

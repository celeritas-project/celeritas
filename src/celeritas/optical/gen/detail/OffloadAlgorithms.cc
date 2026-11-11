//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadAlgorithms.cc
//---------------------------------------------------------------------------//
#include "OffloadAlgorithms.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Append the valid distributions created in this step to the buffer.
 *
 * The distributions written by each track slot past the buffered data are
 * stably compacted, the buffer size is increased by the number of valid
 * distributions, and their photons are added to the number of pending tracks.
 */
void append_distributions(
    optical::CoreState<MemSpace::host>& state,
    ItemsRef<GeneratorDistributionData, MemSpace::host> const& buffer,
    ObserverPtr<GeneratorStateCounters, MemSpace::host> counters,
    size_type num_slots,
    StreamId)
{
    CELER_EXPECT(counters);

    auto& buffer_size = counters->current.buffer_size;
    CELER_EXPECT(buffer_size + num_slots <= buffer.size());

    auto* start = buffer.data().get() + buffer_size;
    auto* stop = std::remove_if(start, start + num_slots, LogicalNot{});

    size_type count = 0;
    for (auto* d = start; d != stop; ++d)
    {
        count += d->num_photons;
    }
    buffer_size += stop - start;

    // If there are more photons, update the number of pending optical photons
    if (count > 0)
    {
        state.add_pending(count);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

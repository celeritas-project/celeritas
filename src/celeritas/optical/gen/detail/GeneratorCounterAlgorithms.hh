//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorCounterAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/ObserverPtr.hh"
#include "celeritas/phys/GeneratorCounters.hh"
#include "celeritas/track/CounterAlgorithms.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
template<MemSpace M>
using GenCountersObserver = ObserverPtr<GeneratorStateCounters, M>;

//---------------------------------------------------------------------------//
/*!
 * Get a pointer to the current buffer size in the counters' memory space.
 *
 * This only computes the address of the member and does not dereference the
 * counters, so it can be used on the host for device counters.
 */
template<MemSpace M>
inline ObserverPtr<size_type, M> buffer_size_ptr(
    GenCountersObserver<M> counters)
{
    CELER_EXPECT(counters);
    auto* ptr = static_cast<GeneratorStateCounters*>(counters);
    return ObserverPtr<size_type, M>{&ptr->current.buffer_size};
}

//---------------------------------------------------------------------------//
// Assign the current generator counters
void assign_counters(
    GenCountersObserver<MemSpace::host>, GeneratorCounters<> const&, StreamId);
void assign_counters(GenCountersObserver<MemSpace::device>,
                     GeneratorCounters<> const&,
                     StreamId);

// Clear the accumulated increments
void clear_accum(GenCountersObserver<MemSpace::host>, StreamId);
void clear_accum(GenCountersObserver<MemSpace::device>, StreamId);

// Clear the number of generated photons
void clear_generated(GenCountersObserver<MemSpace::host>, StreamId);
void clear_generated(GenCountersObserver<MemSpace::device>, StreamId);

// Count the buffered distributions as queued
void accumulate_buffer(GenCountersObserver<MemSpace::host>, StreamId);
void accumulate_buffer(GenCountersObserver<MemSpace::device>, StreamId);

// Set the buffer size from a compaction, counting appended distributions
void set_buffer_size(GenCountersObserver<MemSpace::host>,
                     ObserverPtr<size_type const, MemSpace::host>,
                     StreamId);
void set_buffer_size(GenCountersObserver<MemSpace::device>,
                     ObserverPtr<size_type const, MemSpace::device>,
                     StreamId);

// Set the number of pending photons from a total, counting the new photons
void update_pending(GenCountersObserver<MemSpace::host>,
                    ObserverPtr<size_type const, MemSpace::host>,
                    StreamId);
void update_pending(GenCountersObserver<MemSpace::device>,
                    ObserverPtr<size_type const, MemSpace::device>,
                    StreamId);

// Update the generator and core counters after generating photons
void update_counters(CoreStateCounterRef<MemSpace::host> const&,
                     GenCountersObserver<MemSpace::host>,
                     size_type,
                     StreamId);
void update_counters(CoreStateCounterRef<MemSpace::device> const&,
                     GenCountersObserver<MemSpace::device>,
                     size_type,
                     StreamId);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas

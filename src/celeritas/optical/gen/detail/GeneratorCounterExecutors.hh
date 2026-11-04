//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorCounterExecutors.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/phys/GeneratorCounters.hh"
#include "celeritas/track/CoreStateCounters.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
using GenCountersPtr = ObserverPtr<GeneratorStateCounters, MemSpace::native>;
using CoreCountersPtr = ObserverPtr<CoreStateCounters, MemSpace::native>;

//---------------------------------------------------------------------------//
/*!
 * Assign the current generator counters from a value sent from the host.
 */
struct AssignGeneratorCountersExecutor
{
    GenCountersPtr counters;
    GeneratorCounters<> value;

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        CELER_EXPECT(tid == ThreadId{0});
        counters->current = value;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Clear the accumulated increments after they are read on the host.
 */
struct ClearAccumExecutor
{
    GenCountersPtr counters;

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        CELER_EXPECT(tid == ThreadId{0});
        counters->accum = {};
    }
};

//---------------------------------------------------------------------------//
/*!
 * Clear the number of generated photons at the start of a transport loop.
 */
struct ClearGeneratedExecutor
{
    GenCountersPtr counters;

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        CELER_EXPECT(tid == ThreadId{0});
        counters->current.num_generated = 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Count the buffered distributions as queued.
 *
 * This is used when a generator starts generating from a buffer that only
 * contains newly added distributions.
 */
struct AccumulateBufferExecutor
{
    GenCountersPtr counters;

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        CELER_EXPECT(tid == ThreadId{0});
        counters->accum.buffer_size += counters->current.buffer_size;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Set the buffer size from the number of distributions selected in place.
 *
 * Any increase in the buffer size comes from distributions appended past the
 * compacted buffer, which are counted as queued.
 */
struct SetBufferSizeExecutor
{
    GenCountersPtr counters;
    ObserverPtr<size_type const, MemSpace::native> num_selected;

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        CELER_EXPECT(tid == ThreadId{0});
        CELER_EXPECT(num_selected);

        auto& current = counters->current;
        size_type buffer_size = *num_selected;
        if (buffer_size > current.buffer_size)
        {
            counters->accum.buffer_size += buffer_size - current.buffer_size;
        }
        current.buffer_size = buffer_size;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Set the number of pending photons from the total of a prefix sum.
 *
 * The increase in the number of pending photons is counted as queued. The
 * core state's pending count is not changed, since the photons are added to
 * it when their distributions are created or inserted.
 */
struct UpdatePendingExecutor
{
    GenCountersPtr counters;
    ObserverPtr<size_type const, MemSpace::native> total;

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        CELER_EXPECT(tid == ThreadId{0});
        CELER_EXPECT(total);

        auto& current = counters->current;
        size_type num_pending = *total;
        CELER_ASSERT(num_pending >= current.num_pending);
        counters->accum.num_pending += num_pending - current.num_pending;
        current.num_pending = num_pending;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Update the generator and core state counters after generating photons.
 *
 * The number of tracks generated in this step is the smaller of the number of
 * vacancies and the number of photons pending in the generator. When all the
 * pending photons have been generated, the buffered data is cleared. The
 * number of generated photons is kept until the start of the next transport
 * loop.
 */
struct UpdateGeneratorCountersExecutor
{
    CoreCountersPtr core_counters;
    GenCountersPtr counters;
    size_type state_size{};

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        CELER_EXPECT(tid == ThreadId{0});
        CELER_EXPECT(state_size >= core_counters->num_vacancies);

        auto& core = *core_counters;
        auto& current = counters->current;

        size_type num_gen = min(core.num_vacancies, current.num_pending);

        // Update the optical core state counters
        core.num_pending -= num_gen;
        core.num_generated += num_gen;
        core.num_vacancies -= num_gen;

        // Update the generator counters
        current.num_pending -= num_gen;
        current.num_generated += num_gen;
        counters->accum.num_generated += num_gen;

        if (current.num_pending == 0)
        {
            // All buffered data has been consumed
            current.buffer_size = 0;
        }

        // Update the number of active tracks. This must be done even if no
        // new tracks were generated
        core.num_active = state_size - core.num_vacancies;
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas

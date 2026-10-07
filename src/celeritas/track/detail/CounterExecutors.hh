//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/CounterExecutors.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ThreadId.hh"

#include "../CoreStateCounters.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Clear the num_generated, num_cut, and num_errored counters.
 *
 * This is called for the core loop immediately before a step iteration is
 * started in Stepper<M>::async() .
 *
 * \todo Instead of manually calling there, should it be part of the pre-init
 * action?
 *
 * \todo For the main EM loop, these are reset once per *step*. For the optical
 * loop, they are reset once per *transport*.
 */
struct ResetCountersExecutor
{
    //// DATA ////

    ObserverPtr<CoreStateCounters, MemSpace::native> counters;

    //// FUNCTIONS ////

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        CELER_EXPECT(tid == ThreadId{0});

        counters->num_generated = 0;
        counters->num_cut = 0;
        counters->num_errored = 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Add to the num_pending counter based on photons from buffered
 * optical distribution data.
 *
 * This is a \em thread executor suitable for \c KernelLauncher or \c
 * launch_kernel with a \c num_threads value of 1.
 *
 * \note This is templated on the counter type (allowed to be integer *or*
 * pointer to integer) to allow the counter value to live on the device rather
 * than being copied through the kernel launch.
 */
template<typename CounterType>
struct AddPendingExecutor
{
    //// DATA ////

    ObserverPtr<CoreStateCounters, MemSpace::native> counters;
    CounterType num_photons;

    //// FUNCTIONS ////

    // Add number of primaries waiting to be generated
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        CELER_EXPECT(tid == ThreadId{0});

        size_type temp_num_photons{};
        if constexpr (std::is_integral_v<CounterType>)
        {
            // Copied via kernel launch
            temp_num_photons = num_photons;
        }
        else
        {
            // Lives elsewhere in this memspace
            temp_num_photons = *num_photons;
        }
        counters->num_pending += temp_num_photons;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Update track initializer counters after processing primaries.
 *
 * This uses a primary count sent from the host via the kernel launch, so the
 * number of primaries must be positive.
 */
struct AddPrimariesCountExecutor
{
    //// DATA ////

    ObserverPtr<CoreStateCounters, MemSpace::native> counters;
    size_type num_primaries{};

    //// FUNCTIONS ////

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        CELER_EXPECT(tid == ThreadId{0});
        CELER_EXPECT(num_primaries > 0);

        counters->num_initializers += num_primaries;
        counters->num_generated += num_primaries;
        counters->num_pending = 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Update num_active state counter based on the number of vacancies.
 */
struct UpdateActiveExecutor
{
    //// DATA ////

    ObserverPtr<CoreStateCounters, MemSpace::native> counters;
    size_type state_size{};

    //// FUNCTIONS ////

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        CELER_EXPECT(tid == ThreadId{0});  // single thread kernel
        CELER_EXPECT(state_size >= counters->num_vacancies);

        size_type num_new_tracks
            = min(counters->num_vacancies, counters->num_initializers);

        // Update initializers/vacancies
        counters->num_initializers -= num_new_tracks;
        counters->num_vacancies -= num_new_tracks;

        // Store number of active tracks at the start of the loop
        counters->num_active = state_size - counters->num_vacancies;

        CELER_ENSURE(counters->num_active <= state_size);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Update the num_alive counter based on the number of photons that are still
 * alive after compacting vacancies.
 *
 * (Note this is only used in optical photon loop, not core loop.)
 */
struct UpdateAliveExecutor
{
    //// DATA ////

    ObserverPtr<CoreStateCounters, MemSpace::native> counters;
    size_type state_size;

    //// FUNCTIONS ////

    // Update number of photons that are still alive
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        CELER_EXPECT(tid == ThreadId{0});  // single thread kernel
        CELER_EXPECT(state_size >= counters->num_vacancies);

        counters->num_alive = state_size - counters->num_vacancies;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Update the num_secondaries and num_initializers and then update the number
 * of alive tracks to include these secondaries.
 *
 * The last entry in the secondary counts array holds the exclusive sum, which
 * is the number of secondaries.
 */
struct UpdateSecondariesCountExecutor
{
    //// DATA ////

    ObserverPtr<CoreStateCounters, MemSpace::native> counters;
    ObserverPtr<size_type, MemSpace::native> secondary_count;
    size_type state_size{};

    //// FUNCTIONS ////

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        CELER_EXPECT(tid == ThreadId{0});  // single thread kernel
        CELER_EXPECT(secondary_count);
        CELER_EXPECT(state_size >= counters->num_vacancies);

        size_type num_secondaries = *secondary_count;
        counters->num_secondaries = num_secondaries;
        counters->num_initializers += num_secondaries;
        counters->num_alive = state_size - counters->num_vacancies;
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/UpdatePendingExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/optical/CoreTrackView.hh"

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
 * Update the num_pending counter based on the generated photons from buffered
 * optical distribution data.
 */
template<typename CounterType>
struct UpdatePendingExecutor
{
    //// DATA ////

    ObserverPtr<CoreStateCounters, MemSpace::native> counters;
    CounterType num_photons;

    //// FUNCTIONS ////

    // Update number of primaries waiting to be generated
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
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
}  // namespace detail
}  // namespace optical
}  // namespace celeritas

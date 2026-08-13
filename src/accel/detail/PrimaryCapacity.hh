//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/PrimaryCapacity.hh
//! \sa PrimaryCapacity.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
struct PrimaryCapacityInput
{
    size_type track_slots{};
    size_type initializer_capacity{};
    size_type secondary_capacity{};
    size_type queued{};
    size_type alive{};
};

//---------------------------------------------------------------------------//
/*!
 * Calculate how many primaries can be added without reducing secondary space.
 *
 * Primaries must first fit alongside existing initializers. At the start of
 * the step, queued initializers and primaries fill vacant track slots. Any
 * initializers left after that must preserve the usable secondary capacity.
 */
inline size_type calc_primary_capacity(PrimaryCapacityInput const& input)
{
    CELER_EXPECT(input.queued <= input.initializer_capacity);
    CELER_EXPECT(input.alive <= input.track_slots);

    size_type const available_before_init = input.initializer_capacity
                                            - input.queued;
    size_type const vacancies = input.track_slots - input.alive;
    size_type const unfilled_vacancies
        = vacancies > input.queued ? vacancies - input.queued : 0;
    if (unfilled_vacancies >= available_before_init)
    {
        // Every admitted primary will be consumed during initialization.
        return available_before_init;
    }

    size_type const queued_after_init
        = input.queued > vacancies ? input.queued - vacancies : 0;
    size_type const secondary_reserve
        = std::min(input.secondary_capacity, input.initializer_capacity);
    size_type const queue_capacity = input.initializer_capacity
                                     - secondary_reserve;
    size_type const available_after_init = queued_after_init < queue_capacity
                                               ? queue_capacity
                                                     - queued_after_init
                                               : 0;

    // Limit the addition before summing to avoid overflowing size_type.
    return unfilled_vacancies
           + std::min(available_after_init,
                      available_before_init - unfilled_vacancies);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

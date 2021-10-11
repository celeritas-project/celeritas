//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitializeTracks.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/NumericLimits.hh"
#include "base/Span.hh"
#include "physics/base/Primary.hh"
#include "sim/TrackInterface.hh"
#include "sim/TrackInitInterface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Invalid index flag
CELER_CONSTEXPR_FUNCTION size_type flag_id()
{
    return numeric_limits<size_type>::max();
}

//---------------------------------------------------------------------------//
// Get the thread ID of the last element
inline CELER_FUNCTION ThreadId from_back(size_type size, ThreadId cur_thread);

//---------------------------------------------------------------------------//
// Initialize the track states on device.
void init_tracks(const ParamsDeviceRef&         params,
                 const StateDeviceRef&          states,
                 const TrackInitStateDeviceRef& inits);

void init_tracks(const ParamsHostRef&         params,
                 const StateHostRef&          states,
                 const TrackInitStateHostRef& inits);

//---------------------------------------------------------------------------//
// Identify which tracks are still alive and count the number of secondaries
// that survived cutoffs for each interaction.
void locate_alive(const ParamsDeviceRef&         params,
                  const StateDeviceRef&          states,
                  const TrackInitStateDeviceRef& inits);

void locate_alive(const ParamsHostRef&         params,
                  const StateHostRef&          states,
                  const TrackInitStateHostRef& inits);

//---------------------------------------------------------------------------//
// Create track initializers on device from primary particles
void process_primaries(Span<const Primary>            primaries,
                       const TrackInitStateDeviceRef& inits);

void process_primaries(Span<const Primary>          primaries,
                       const TrackInitStateHostRef& inits);

//---------------------------------------------------------------------------//
// Create track initializers on device from secondary particles.
void process_secondaries(const ParamsDeviceRef&         params,
                         const StateDeviceRef&          states,
                         const TrackInitStateDeviceRef& inits);

void process_secondaries(const ParamsHostRef&         params,
                         const StateHostRef&          states,
                         const TrackInitStateHostRef& inits);

//---------------------------------------------------------------------------//
// Remove all elements in the vacancy vector that were flagged as alive
template<MemSpace M>
size_type remove_if_alive(Span<size_type> vacancies);

template<>
size_type remove_if_alive<MemSpace::host>(Span<size_type> vacancies);
template<>
size_type remove_if_alive<MemSpace::device>(Span<size_type> vacancies);

//---------------------------------------------------------------------------//
// Sum the total number of surviving secondaries.
template<MemSpace M>
size_type reduce_counts(Span<size_type> counts);

template<>
size_type reduce_counts<MemSpace::host>(Span<size_type> counts);
template<>
size_type reduce_counts<MemSpace::device>(Span<size_type> counts);

//---------------------------------------------------------------------------//
// Calculate the exclusive prefix sum of the number of surviving secondaries
template<MemSpace M>
void exclusive_scan_counts(Span<size_type> counts);

template<>
void exclusive_scan_counts<MemSpace::host>(Span<size_type> counts);
template<>
void exclusive_scan_counts<MemSpace::device>(Span<size_type> counts);

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get the thread ID of the last element.
 */
CELER_FUNCTION ThreadId from_back(size_type size, ThreadId cur_thread)
{
    CELER_EXPECT(cur_thread.get() + 1 <= size);
    return ThreadId{size - cur_thread.get() - 1};
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitializeTracks.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "base/Span.hh"
#include "physics/base/Primary.hh"
#include "sim/TrackData.hh"
#include "sim/TrackInitData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Initialize track states
void init_tracks(const ParamsDeviceRef&         params,
                 const StateDeviceRef&          states,
                 const TrackInitStateDeviceRef& data);
void init_tracks(const ParamsHostRef&         params,
                 const StateHostRef&          states,
                 const TrackInitStateHostRef& data);

//---------------------------------------------------------------------------//
// Identify which tracks are alive and count secondaries created
void locate_alive(const ParamsDeviceRef&         params,
                  const StateDeviceRef&          states,
                  const TrackInitStateDeviceRef& data);
void locate_alive(const ParamsHostRef&         params,
                  const StateHostRef&          states,
                  const TrackInitStateHostRef& data);

//---------------------------------------------------------------------------//
// Create track initializers from primary particles
void process_primaries(Span<const Primary>            primaries,
                       const TrackInitStateDeviceRef& data);
void process_primaries(Span<const Primary>          primaries,
                       const TrackInitStateHostRef& data);

//---------------------------------------------------------------------------//
// Create track initializers from secondary particles
void process_secondaries(const ParamsDeviceRef&         params,
                         const StateDeviceRef&          states,
                         const TrackInitStateDeviceRef& data);
void process_secondaries(const ParamsHostRef&         params,
                         const StateHostRef&          states,
                         const TrackInitStateHostRef& data);

//---------------------------------------------------------------------------//
// Remove all elements in the vacancy vector that were flagged as alive
template<MemSpace M>
size_type remove_if_alive(Span<size_type> vacancies);

template<>
size_type remove_if_alive<MemSpace::host>(Span<size_type> vacancies);
template<>
size_type remove_if_alive<MemSpace::device>(Span<size_type> vacancies);

//---------------------------------------------------------------------------//
// Calculate the exclusive prefix sum of the number of surviving secondaries
template<MemSpace M>
size_type exclusive_scan_counts(Span<size_type> counts);

template<>
size_type exclusive_scan_counts<MemSpace::host>(Span<size_type> counts);
template<>
size_type exclusive_scan_counts<MemSpace::device>(Span<size_type> counts);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELERITAS_USE_CUDA
inline void init_tracks(const ParamsDeviceRef&,
                        const StateDeviceRef&,
                        const TrackInitStateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA");
}

inline void locate_alive(const ParamsDeviceRef&,
                         const StateDeviceRef&,
                         const TrackInitStateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA");
}

inline void
process_primaries(Span<const Primary>, const TrackInitStateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA");
}

inline void process_secondaries(const ParamsDeviceRef&,
                                const StateDeviceRef&,
                                const TrackInitStateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA");
}

template<>
size_type remove_if_alive<MemSpace::device>(Span<size_type>)
{
    CELER_NOT_CONFIGURED("CUDA");
}

template<>
size_type exclusive_scan_counts<MemSpace::device>(Span<size_type>)
{
    CELER_NOT_CONFIGURED("CUDA");
}

#endif
//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

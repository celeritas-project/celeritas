//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitializeTracks.nocuda.cc
//---------------------------------------------------------------------------//
#include "InitializeTracks.hh"

#include "base/Assert.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
void init_tracks(const ParamsDeviceRef&,
                 const StateDeviceRef&,
                 const TrackInitStateDeviceRef&)
{
    CELER_ASSERT_UNREACHABLE();
}

void locate_alive(const ParamsDeviceRef&,
                  const StateDeviceRef&,
                  const TrackInitStateDeviceRef&)
{
    CELER_ASSERT_UNREACHABLE();
}

void process_primaries(Span<const Primary>, const TrackInitStateDeviceRef&)
{
    CELER_ASSERT_UNREACHABLE();
}

void process_secondaries(const ParamsDeviceRef&,
                         const StateDeviceRef&,
                         const TrackInitStateDeviceRef&)
{
    CELER_ASSERT_UNREACHABLE();
}

size_type remove_if_alive(Span<size_type>)
{
    CELER_ASSERT_UNREACHABLE();
}

size_type reduce_counts(Span<size_type>)
{
    CELER_ASSERT_UNREACHABLE();
}

void exclusive_scan_counts(Span<size_type>)
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

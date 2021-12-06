//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInitUtils.cc
//---------------------------------------------------------------------------//
#include "TrackInitUtils.hh"

#include "base/Algorithms.hh"
#include "base/DeviceVector.hh"
#include "sim/TrackInitData.hh"
#include "detail/InitializeTracks.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//!@{
/*!
 * Create track initializers on device from primary particles.
 *
 * This creates the maximum possible number of track initializers on device
 * from host primaries (either the number of host primaries that have not yet
 * been initialized on device or the size of the available storage in the track
 * initializer vector, whichever is smaller).
 */
void extend_from_primaries(
    const TrackInitParamsData<Ownership::const_reference, MemSpace::host>& params,
    TrackInitStateDeviceVal* data)
{
    CELER_EXPECT(params);
    CELER_EXPECT(data && *data);

    // Number of primaries to copy to device
    auto count = min(data->initializers.capacity() - data->initializers.size(),
                     data->num_primaries);
    if (count)
    {
        data->initializers.resize(data->initializers.size() + count);

        // Allocate memory on device and copy primaries
        DeviceVector<Primary> primaries(count);
        primaries.copy_to_device(params.primaries[ItemRange<Primary>(
            ItemId<Primary>(data->num_primaries - count),
            ItemId<Primary>(data->num_primaries))]);
        data->num_primaries -= count;

        // Launch a kernel to create track initializers from primaries
        detail::process_primaries(primaries.device_ref(), make_ref(*data));
    }
}

void extend_from_primaries(
    const TrackInitParamsData<Ownership::const_reference, MemSpace::host>& params,
    TrackInitStateHostVal* data)
{
    CELER_EXPECT(params);
    CELER_EXPECT(data && *data);

    // Number of primaries to copy to device
    auto count = min(data->initializers.capacity() - data->initializers.size(),
                     data->num_primaries);
    if (count)
    {
        data->initializers.resize(data->initializers.size() + count);

        // Allocate memory on device and copy primaries
        auto primaries = params.primaries[ItemRange<Primary>(
            ItemId<Primary>(data->num_primaries - count),
            ItemId<Primary>(data->num_primaries))];
        data->num_primaries -= count;

        // Launch a kernel to create track initializers from primaries
        detail::process_primaries(primaries, make_ref(*data));
    }
}
//!@}
//---------------------------------------------------------------------------//
} // namespace celeritas

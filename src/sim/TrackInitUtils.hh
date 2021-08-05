//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInitUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "sim/TrackInitInterface.hh"
#include "sim/TrackInterface.hh"
#include "sim/detail/InitializeTracks.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
// Create track initializers on device from primary particles
void extend_from_primaries(const TrackInitParamsHostRef& params,
                           TrackInitStateDeviceVal*      data);

// Create track initializers on host from primary particles
void extend_from_primaries(const TrackInitParamsHostRef& params,
                           TrackInitStateHostVal*        data);

// Create track initializers on device from secondary particles.
void extend_from_secondaries(const ParamsDeviceRef&   params,
                             const StateDeviceRef&    states,
                             TrackInitStateDeviceVal* data);

//---------------------------------------------------------------------------//
/*!
 * Initialize track states
 *
 * Tracks created from secondaries produced in this step will have the geometry
 * state copied over from the parent instead of initialized from the position.
 * If there are more empty slots than new secondaries, they will be filled by
 * any track initializers remaining from previous steps using the position.
 */
template<MemSpace M>
inline void
initialize_tracks(const ParamsData<Ownership::const_reference, M>& params,
                  const StateData<Ownership::reference, M>&        states,
                  TrackInitStateData<Ownership::value, M>*         data)
{
    CELER_EXPECT(params);
    CELER_EXPECT(states);
    CELER_EXPECT(data && *data);

    // The number of new tracks to initialize is the smaller of the number of
    // empty slots in the track vector and the number of track initializers
    size_type num_tracks
        = std::min(data->vacancies.size(), data->initializers.size());
    if (num_tracks > 0)
    {
        // Launch a kernel to initialize tracks on device
        detail::init_tracks(params, states, make_ref(*data));
        data->initializers.resize(data->initializers.size() - num_tracks);
        data->vacancies.resize(data->vacancies.size() - num_tracks);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas

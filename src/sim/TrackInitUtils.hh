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

// Initialize track states on device.
void initialize_tracks(const ParamsDeviceRef&   params,
                       const StateDeviceRef&    states,
                       TrackInitStateDeviceVal* data);

//---------------------------------------------------------------------------//
} // namespace celeritas

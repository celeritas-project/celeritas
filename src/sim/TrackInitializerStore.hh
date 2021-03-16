//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInitializerStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/DeviceVector.hh"
#include "ParamStore.hh"
#include "StateStore.hh"
#include "TrackInitializerInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage device data for track initializers.
 */
class TrackInitializerStore
{
  public:
    // Construct with the number of tracks, the maximum number of track
    // initializers to store on device, and the primary particles
    explicit TrackInitializerStore(size_type            num_tracks,
                                   size_type            capacity,
                                   std::vector<Primary> primaries);

    // Get a view to the managed data
    TrackInitializerPointers device_pointers();

    //! Number of track initializers
    size_type size() const { return initializers_.size(); }

    //! Number of empty track slots on device
    size_type num_vacancies() const { return vacancies_.size(); }

    //! Number of primary particles left to be initialized on device
    size_type num_primaries() const { return primaries_.size(); }

    // Create track initializers on device from primary particles
    void extend_from_primaries();

    // Create track initializers on device from secondary particles.
    void extend_from_secondaries(StateStore* states, ParamStore* params);

    // Initialize track states on device.
    void initialize_tracks(StateStore* states, ParamStore* params);

  private:
    // Track initializers created from primaries or secondaries
    DeviceVector<TrackInitializer> initializers_;

    // Thread ID of the secondary's parent
    DeviceVector<size_type> parent_;

    // Index of empty slots in track vector
    DeviceVector<size_type> vacancies_;

    // Number of surviving secondaries produced in each interaction
    DeviceVector<size_type> secondary_counts_;

    // Track ID counter for each event
    DeviceVector<TrackId::size_type> track_counter_;

    // Host-side primary particles
    std::vector<Primary> primaries_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

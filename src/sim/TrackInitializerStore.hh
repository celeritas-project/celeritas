//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInitializerStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/CollectionBuilder.hh"
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
    TrackInitializerDeviceRef device_pointers() { return make_ref(data_); }

    //! Number of track initializers
    size_type size() const { return data_.initializers.size(); }

    //! Number of empty track slots on device
    size_type num_vacancies() const { return data_.vacancies.size(); }

    //! Number of primary particles left to be initialized on device
    size_type num_primaries() const { return primaries_.size(); }

    // Create track initializers on device from primary particles
    void extend_from_primaries();

    // Create track initializers on device from secondary particles.
    void extend_from_secondaries(StateStore* states, ParamStore* params);

    // Initialize track states on device.
    void initialize_tracks(StateStore* states, ParamStore* params);

  private:
    // Host-side primary particles
    std::vector<Primary> primaries_;

    // Device-side storage
    TrackInitializerData<Ownership::value, MemSpace::device> data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

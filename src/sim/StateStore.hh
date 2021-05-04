//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StateStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/DeviceVector.hh"
#include "geometry/GeoParams.hh"
#include "random/RngInterface.hh"
#include "SimInterface.hh"
#include "TrackInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage device data for tracks.
 */
class StateStore
{
  public:
    //!@{
    //! Type aliases
    using SPConstGeo = std::shared_ptr<const GeoParams>;
    //!@}

    //! Construction arguments
    struct Input
    {
        size_type    num_tracks;
        SPConstGeo   geo;
        unsigned int host_seed = 12345u;
    };

  public:
    // Construct with the track state input
    explicit StateStore(const Input& inp);

    //! Get the total number of tracks
    size_type size() const { return particle_states_.size(); }

    // Get a view to the managed data
    StatePointers device_pointers();

  private:
    template<template<Ownership, MemSpace> class S>
    using DeviceVal = S<Ownership::value, MemSpace::device>;

    // XXX Unify these guys and possibly remove this state store?
    DeviceVal<ParticleStateData> particle_states_;
    DeviceVal<RngStateData>      rng_states_;
    DeviceVal<GeoStateData>      geo_states_;
    DeviceVal<SimStateData>      sim_states_;

    DeviceVector<Interaction> interactions_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

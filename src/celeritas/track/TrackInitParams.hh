//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInitParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "TrackInitData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage persistent track initializer data.
 */
class TrackInitParams final : public ParamsDataInterface<TrackInitParamsData>
{
  public:
    //! Track initializer construction arguments
    struct Input
    {
        size_type capacity{};  //!< Max number of initializers
        size_type max_events{};  //!< Max number of events that can be run
        TrackOrder track_order{TrackOrder::unsorted};  //!< How to sort tracks
    };

  public:
    // Construct with capacity and number of events
    explicit TrackInitParams(Input const&);

    //! Event number cannot exceed this value
    size_type max_events() const { return host_ref().max_events; }

    //! Access primaries for contructing track initializer states
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<TrackInitParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

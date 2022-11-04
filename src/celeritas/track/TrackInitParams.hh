//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInitParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"

#include "TrackInitData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage persistent track initializer data.
 */
class TrackInitParams
{
  public:
    //!@{
    //! References to constructed data
    using HostRef   = HostCRef<TrackInitParamsData>;
    using DeviceRef = DeviceCRef<TrackInitParamsData>;
    //!@}

    //! Track initializer construction arguments
    struct Input
    {
        size_type capacity;   //!< Max number of initializers
        size_type max_events; //!< Max number of events that can be run
    };

  public:
    // Construct with capacity and number of events
    explicit TrackInitParams(const Input&);

    //! Access primaries for contructing track initializer states
    const HostRef& host_ref() const { return data_.host(); }

    //! Access data on device
    const DeviceRef& device_ref() const { return data_.device(); }

  private:
    // Host/device storage and reference
    CollectionMirror<TrackInitParamsData> data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

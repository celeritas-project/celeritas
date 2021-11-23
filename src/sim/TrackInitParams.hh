//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInitParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "base/Types.hh"
#include "physics/base/Primary.hh"
#include "TrackInitData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage persistent track initializer data.
 *
 * Primary particles are stored on the host and only copied to device when they
 * are needed to initialize new tracks.
 */
class TrackInitParams
{
  public:
    //!@{
    //! References to constructed data
    using HostRef
        = TrackInitParamsData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = TrackInitParamsData<Ownership::const_reference, MemSpace::device>;
    //!@}

    //! Track initializer construction arguments
    struct Input
    {
        std::vector<Primary> primaries;
        size_type            storage_factor;
    };

  public:
    // Construct with primaries and storage factor
    explicit TrackInitParams(const Input&);

    //! Access primaries for contructing track initializer states
    const HostRef& host_ref() const { return host_ref_; }

    //! Access data on device
    const DeviceRef& device_ref() const { return device_ref_; }

  private:
    using HostValue = TrackInitParamsData<Ownership::value, MemSpace::host>;

    HostValue host_value_;
    HostRef   host_ref_;
    DeviceRef device_ref_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

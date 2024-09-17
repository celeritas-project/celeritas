//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/TrackInitParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "TrackInitData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Manage persistent track initializer data.
 */
class TrackInitParams final : public ParamsDataInterface<TrackInitParamsData>
{
  public:
    // Construct with capacity
    explicit TrackInitParams(size_type capacity);

    //! Access data on host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<TrackInitParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

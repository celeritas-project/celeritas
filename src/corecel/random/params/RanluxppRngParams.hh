//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/params/RanluxppRngParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/random/data/RanluxppRngData.hh"
#include "corecel/random/data/RanluxppTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared data for Ranluxpp pseudo-random number generator.
 */
class RanluxppRngParams final
    : public ParamsDataInterface<RanluxppRngParamsData>
{
  public:
    // Construct with seed
    explicit RanluxppRngParams(RanluxppUInt seed);

    //! Access rng params data on host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access rng params data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    /// DATA ///
    // Host/device storage and reference
    CollectionMirror<RanluxppRngParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

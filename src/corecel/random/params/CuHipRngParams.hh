//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/params/CuHipRngParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/random/data/CuHipRngData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage random number generation.
 *
 * Currently this just constructs a local seed number but should be extended to
 * handle RNG setup across multiple MPI processes.
 */
class CuHipRngParams final : public ParamsDataInterface<CuHipRngParamsData>
{
  public:
    // Construct with seed
    explicit CuHipRngParams(unsigned int seed);

    //! Access RNG properties for constructing RNG state
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<CuHipRngParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

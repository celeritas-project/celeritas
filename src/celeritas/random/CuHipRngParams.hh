//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/CuHipRngParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "CuHipRngData.hh"

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

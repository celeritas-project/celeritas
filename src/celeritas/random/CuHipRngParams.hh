//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/CuHipRngParams.hh
//---------------------------------------------------------------------------//
#pragma once

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
class CuHipRngParams
{
  public:
    //!@{
    //! References to constructed data
    using HostRef = HostCRef<CuHipRngParamsData>;
    using DeviceRef = DeviceCRef<CuHipRngParamsData>;
    //!@}

  public:
    // Construct with seed
    explicit inline CuHipRngParams(unsigned int seed);

    //! Access RNG properties for constructing RNG state
    HostRef const& host_ref() const { return host_ref_; }

    //! Access data on device
    DeviceRef const& device_ref() const { return device_ref_; }

  private:
    HostRef host_ref_;
    DeviceRef device_ref_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with seed.
 */
CuHipRngParams::CuHipRngParams(unsigned int seed)
{
    host_ref_.seed = seed;
}

}  // namespace celeritas

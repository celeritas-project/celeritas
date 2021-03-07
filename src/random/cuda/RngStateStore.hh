//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStateStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/DeviceVector.hh"
#include "base/Types.hh"
#include "RngInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage ownership of on-device random number generator.
 */
class RngStateStore
{
  public:
    // Empty constructor
    RngStateStore() = default;

    // Construct with the number of RNG states
    explicit RngStateStore(size_type size, unsigned int host_seed = 12345u);

    //! Number of states
    size_type size() const { return data_.size(); }

    // Access pointers to on-device memory
    RngStatePointers device_pointers();

  private:
    // Stored RNG states on device
    DeviceVector<RngState> data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DetectorStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "base/Types.hh"
#include "base/StackAllocatorStore.hh"
#include "base/DeviceVector.hh"
#include "base/UniformGrid.hh"
#include "DetectorPointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Storage for a simple 'event-wise' detector.
 */
class DetectorStore
{
  public:
    // Construct with the given capacity for hits
    explicit DetectorStore(size_type                  buffer_capacity,
                           const UniformGrid::Params& grid);

    // Get reference to on-device data
    DetectorPointers device_pointers();

    // Launch a kernel to bin the buffer into the grid
    void bin_buffer();

    // Finalize, copy to CPU, and reset, normalizing with the given value
    std::vector<real_type> finalize(real_type norm);

  private:
    // In-kernel hit buffer
    StackAllocatorStore<Hit> hit_buffer_;
    // Uniform tally grid
    UniformGrid::Params tally_grid_;
    // Tallied data
    DeviceVector<real_type> tally_deposition_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

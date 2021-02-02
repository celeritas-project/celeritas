//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file HostDetectorStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "base/Types.hh"
#include "physics/grid/UniformGrid.hh"
#include "DetectorInterface.hh"
#include "HostStackAllocatorStore.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Host storage for a simple 'event-wise' detector.
 */
class HostDetectorStore
{
  public:
    // Construct with defaults
    HostDetectorStore(size_type                  buffer_capacity,
                      const UniformGridPointers& grid);

    // Get detector data
    DetectorPointers host_pointers();

    // Bin the buffer onto the grid
    void bin_buffer();

    // Finalize the tally result
    std::vector<real_type> finalize(real_type norm);

  private:
    // Host-side hit buffer
    HostStackAllocatorStore<Hit> hit_buffer_;
    // Uniform tally grid
    UniformGridPointers tally_grid_;
    // Tallied data
    std::vector<real_type> tally_deposition_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

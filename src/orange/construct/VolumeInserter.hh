//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VolumeInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"
#include "orange/Types.hh"

#include "../Data.hh"
#include "VolumeInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct volumes on the host.
 *
 * Currently this requires the full volume attribute exported from
 * SCALE-ORANGE, but it can be reworked in the future to calculate attributes
 * (such as number of intersections) at construction time. We can also
 * implement deduplication of the logic references to reduce memory
 * storage and improve locality. Finally, when we add support for multiple
 * universes, we might need to add a surface ID mapping for the volume input,
 * since the face IDs from one universe won't match the stored global face IDs
 * inside Celeritas-ORANGE.
 */
class VolumeInserter
{
  public:
    //!@{
    //! Type aliases
    using SurfData = SurfaceData<Ownership::const_reference, MemSpace::host>;
    using Data = VolumeData<Ownership::value, MemSpace::host>;
    //!@}

  public:
    // Construct from input surface data and targeted volume data
    VolumeInserter(const SurfData& surfaces, Data* volumes);

    // Append a volume
    VolumeId operator()(const VolumeInput& vol_def);

    //! Get the maximum stack depth of any volume definition
    int max_logic_depth() const { return max_logic_depth_; }

  private:
    const SurfData& surface_data_;
    Data* volume_data_{nullptr};
    int   max_logic_depth_{0};
};

//---------------------------------------------------------------------------//
} // namespace celeritas

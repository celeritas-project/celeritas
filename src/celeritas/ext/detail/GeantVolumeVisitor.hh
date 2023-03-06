//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantVolumeVisitor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <vector>

class G4LogicalVolume;

namespace celeritas
{
struct ImportVolume;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Traverse a logical volume hierarchy to record import volumes.
 */
class GeantVolumeVisitor
{
  public:
    // Construct with the unique volume flag
    explicit inline GeantVolumeVisitor(bool unique_volumes);

    // Recurse into the given logical volume
    void visit(G4LogicalVolume const& logical_volume);

    // Transform the map of volumes into a contiguous vector with empty spaces
    std::vector<ImportVolume> build_volume_vector() const;

  private:
    bool unique_volumes_;
    std::map<int, ImportVolume> volids_volumes_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a uniquifying volume flag.
 */
GeantVolumeVisitor::GeantVolumeVisitor(bool unique_volumes)
    : unique_volumes_(unique_volumes)
{
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

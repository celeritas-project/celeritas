//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantVolumeVisitor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <string>
#include <vector>

#include "corecel/io/Label.hh"
#include "celeritas/io/ImportVolume.hh"

class G4LogicalVolume;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Traverse a logical volume hierarchy to record import volumes.
 */
class GeantVolumeVisitor
{
  public:
    // Generate the GDML name for a Geant4 logical volume
    static std::string generate_name(G4LogicalVolume const& logical_volume);

    // Construct with the unique volume flag
    explicit inline GeantVolumeVisitor(bool unique_volumes);

    // Recurse into the given logical volume
    void visit(G4LogicalVolume const& logical_volume);

    // Transform the map of volumes into a contiguous vector with empty spaces
    std::vector<ImportVolume> build_volume_vector() const;

    // Transform the map of volumes into a vector of Labels (for GeantGeo)
    std::vector<Label> build_label_vector() const;

  private:
    bool unique_volumes_;
    std::map<int, ImportVolume> volids_volumes_;

    // Generate the GDML name for a Geant4 logical volume
    static std::pair<std::string, G4LogicalVolume const*>
    generate_name_refl(G4LogicalVolume const& logical_volume);
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

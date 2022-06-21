//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/DetectorConstruction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4VUserDetectorConstruction.hh>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Load the detector geometry from a GDML input file.
 */
class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    // Construct from a GDML filename
    explicit DetectorConstruction(const std::string& gdml_input);

    G4VPhysicalVolume*       Construct() override;
    const G4VPhysicalVolume* get_world_volume() const;

  private:
    std::unique_ptr<G4VPhysicalVolume> phys_vol_world_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

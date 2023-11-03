//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/DetectorConstruction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <memory>
#include <string>
#include <G4VUserDetectorConstruction.hh>

class G4LogicalVolume;
class G4MagneticField;

namespace celeritas
{
class RZMapFieldParams;
class SharedParams;

namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct a detector from a GDML filename set in GlobalSetup.
 */
class DetectorConstruction final : public G4VUserDetectorConstruction
{
  public:
    //!@{
    //! \name Type aliases
    using SPParams = std::shared_ptr<SharedParams>;
    //!@}

  public:
    // Set up global celeritas SD options during construction
    DetectorConstruction(SPParams params);

    G4VPhysicalVolume* Construct() final;
    void ConstructSDandField() final;

  private:
    SPParams params_;

    std::unique_ptr<G4VPhysicalVolume> world_;
    std::multimap<std::string, G4LogicalVolume*> detectors_;

    // Mangetic field
    std::shared_ptr<RZMapFieldParams> field_params_;
    std::shared_ptr<G4MagneticField> mag_field_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

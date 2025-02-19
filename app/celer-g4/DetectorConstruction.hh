//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/DetectorConstruction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <memory>
#include <string>
#include <G4VUserDetectorConstruction.hh>

#include "accel/SetupOptions.hh"

class G4LogicalVolume;
class G4MagneticField;

namespace celeritas
{
//---------------------------------------------------------------------------//
class SharedParams;
class GeantSimpleCalo;

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
    explicit DetectorConstruction(SPParams params);

    G4VPhysicalVolume* Construct() final;
    void ConstructSDandField() final;

  private:
    //// TYPES ////

    using MapDetectors = std::multimap<std::string, G4LogicalVolume*>;
    using AlongStepFactory = SetupOptions::AlongStepFactory;
    using SPMagneticField = std::shared_ptr<G4MagneticField>;
    using SPSimpleCalo = std::shared_ptr<GeantSimpleCalo>;

    struct FieldData
    {
        AlongStepFactory along_step;
        SPMagneticField g4field;
    };

    //// DATA ////

    SPParams params_;

    MapDetectors detectors_;
    SPMagneticField mag_field_;
    std::vector<SPSimpleCalo> simple_calos_;

    //// METHODS ////

    FieldData construct_field() const;

    template<class F>
    void foreach_detector(F&&) const;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

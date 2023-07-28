//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/GlobalSetup.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4ThreeVector.hh>

#include "accel/SetupOptions.hh"

class G4GenericMessenger;

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Global configuration for setting from the UI under "/config".
 */
class GlobalSetup
{
  public:
    // Return non-owning pointer to a singleton
    static GlobalSetup* Instance();

    //!@{
    //! \name Demo setup options
    std::string const& GetGeometryFile() const { return geometry_file_; }
    std::string const& GetEventFile() const { return event_file_; }
    int GetRootBufferSize() const { return root_buffer_size_; }
    bool GetWriteSDHits() const { return write_sd_hits_; }
    bool StripGDMLPointers() const { return strip_gdml_pointers_; }
    std::string const& GetPhysicsList() const { return physics_list_; }
    bool StepDiagnostic() const { return step_diagnostic_; }
    int GetStepDiagnosticBins() const { return step_diagnostic_bins_; }
    //!@}

    //! Get a mutable reference to the setup options for DetectorConstruction
    SDSetupOptions& GetSDSetupOptions() { return options_->sd; }

    //! Get an immutable reference to the setup options
    std::shared_ptr<SetupOptions const> GetSetupOptions() const
    {
        return options_;
    }

    // Set the list of ignored EM process names
    void SetIgnoreProcesses(SetupOptions::VecString ignored);

    //! Set the field to this value (T) along the z axis
    void SetMagFieldZTesla(double f)
    {
        field_ = G4ThreeVector(0, 0, f * CLHEP::tesla);
    }

  private:
    // Private constructor since we're a singleton
    GlobalSetup();
    ~GlobalSetup();

    // Data
    std::shared_ptr<SetupOptions> options_;
    std::string geometry_file_;
    std::string event_file_;
    int root_buffer_size_{128000};
    bool write_sd_hits_{false};
    bool strip_gdml_pointers_{true};
    std::string physics_list_{"FTFP_BERT"};
    bool step_diagnostic_{false};
    int step_diagnostic_bins_{1000};
    G4ThreeVector field_;

    std::unique_ptr<G4GenericMessenger> messenger_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

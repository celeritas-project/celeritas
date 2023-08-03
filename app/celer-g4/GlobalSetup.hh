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

#include "RunInput.hh"

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
    std::string const& GetInputFile() const { return input_file_; }
    std::string const& GetGeometryFile() const { return geometry_file_; }
    std::string const& GetEventFile() const { return input_.event_file; }
    int GetRootBufferSize() const { return root_buffer_size_; }
    bool GetWriteSDHits() const { return write_sd_hits_; }
    bool StripGDMLPointers() const { return strip_gdml_pointers_; }
    PhysicsList GetPhysicsList() const { return input_.physics_list; }
    bool StepDiagnostic() const { return input_.step_diagnostic; }
    int GetStepDiagnosticBins() const { return input_.step_diagnostic_bins; }
    std::string const& GetFieldType() const { return field_type_; }
    std::string const& GetFieldFile() const { return field_file_; }
    G4ThreeVector GetMagFieldZTesla() const { return field_; }
    //!@}

    //! Get a mutable reference to the setup options for DetectorConstruction
    SDSetupOptions& GetSDSetupOptions() { return options_->sd; }

    //! Set an along step factory to the setup options
    void SetAlongStepFactory(SetupOptions::AlongStepFactory factory)
    {
        options_->make_along_step = std::move(factory);
    }

    //! Get an immutable reference to the setup options
    std::shared_ptr<SetupOptions const> GetSetupOptions() const
    {
        return options_;
    }

    //! Get the physics options for the GeantPhysicsList
    GeantPhysicsOptions const& GetPhysicsOptions() const
    {
        return input_.physics_options;
    }

    //! Get the field driver options
    FieldDriverOptions const& GetFieldOptions() const
    {
        return input_.field_options;
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
    std::string input_file_;
    std::string geometry_file_;
    int root_buffer_size_{128000};
    bool write_sd_hits_{false};
    bool strip_gdml_pointers_{true};
    std::string field_type_{"uniform"};
    std::string field_file_;
    G4ThreeVector field_;
    RunInput input_;

    std::unique_ptr<G4GenericMessenger> messenger_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

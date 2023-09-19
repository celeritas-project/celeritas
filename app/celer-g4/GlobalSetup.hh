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

#include "corecel/sys/Stopwatch.hh"
#include "celeritas/ext/Convert.geant.hh"
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
    std::string const& GetGeometryFile() const { return input_.geometry_file; }
    std::string const& GetEventFile() const { return input_.event_file; }
    PrimaryGeneratorOptions const& GetPrimaryGeneratorOptions() const
    {
        return input_.primary_options;
    }
    int GetRootBufferSize() const { return input_.root_buffer_size; }
    bool GetWriteSDHits() const { return input_.write_sd_hits; }
    bool StripGDMLPointers() const { return input_.strip_gdml_pointers; }
    PhysicsListSelection GetPhysicsList() const { return input_.physics_list; }
    GeantPhysicsOptions const& GetPhysicsOptions() const
    {
        return input_.physics_options;
    }
    bool StepDiagnostic() const { return input_.step_diagnostic; }
    int GetStepDiagnosticBins() const { return input_.step_diagnostic_bins; }
    std::string const& GetFieldType() const { return input_.field_type; }
    std::string const& GetFieldFile() const { return input_.field_file; }
    Real3 GetMagFieldZTesla() const { return input_.field; }
    FieldDriverOptions const& GetFieldOptions() const
    {
        return input_.field_options;
    }
    //!@}

    //! Get the number of events
    int GetNumEvents() { return num_events_; }

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

    // Set the list of ignored EM process names
    void SetIgnoreProcesses(SetupOptions::VecString ignored);

    //! Set the field to this value (T) along the z axis
    void SetMagFieldZTesla(double f) { input_.field = Real3{0, 0, f}; }

    // Read input from macro or JSON
    void ReadInput(std::string const& filename);

    // Get the time for setup
    real_type GetSetupTime() { return get_setup_time_(); }

  private:
    // Private constructor since we're a singleton
    GlobalSetup();
    ~GlobalSetup();

    // Data
    std::shared_ptr<SetupOptions> options_;
    RunInput input_;
    Stopwatch get_setup_time_;
    int num_events_{0};

    std::unique_ptr<G4GenericMessenger> messenger_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "geocel/g4/Convert.hh"
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
    //! \name Demo setup options (DEPRECATED: use direct interface to input)
    std::string const& GetGeometryFile() const { return input_.geometry_file; }
    GeantPhysicsOptions const& GetPhysicsOptions() const
    {
        return input_.physics_options;
    }
    bool StepDiagnostic() const { return input_.step_diagnostic; }
    int GetStepDiagnosticBins() const { return input_.step_diagnostic_bins; }
    std::string const& GetFieldType() const { return input_.field_type; }
    std::string const& GetFieldFile() const { return input_.field_file; }
    Real3 GetMagFieldTesla() const { return input_.field; }
    FieldDriverOptions const& GetFieldOptions() const
    {
        return input_.field_options;
    }
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

    // Set the list of ignored EM process names
    void SetIgnoreProcesses(SetupOptions::VecString ignored);

    //! Set the field to this value (T) along the z axis
    void SetMagFieldZTesla(real_type f) { input_.field = Real3{0, 0, f}; }

    // Read input from JSON
    void ReadInput(std::string const& filename);

    // Get the time for setup
    real_type GetSetupTime() { return get_setup_time_(); }

    //// NEW INTERFACE ////

    //! Get setup options
    SetupOptions const& setup_options() const { return *options_; }

    //! Get user input options
    RunInput const& input() const { return input_; }

    //! Whether ROOT I/O for SDs is enabled
    bool root_sd_io() const { return root_sd_io_; }

  private:
    // Private constructor since we're a singleton
    GlobalSetup();
    ~GlobalSetup();

    // Data
    std::shared_ptr<SetupOptions> options_;
    RunInput input_;
    Stopwatch get_setup_time_;
    bool root_sd_io_{false};

    std::unique_ptr<G4GenericMessenger> messenger_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

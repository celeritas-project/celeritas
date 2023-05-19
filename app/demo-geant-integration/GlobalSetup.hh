//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/GlobalSetup.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4ThreeVector.hh>

#include "accel/SetupOptions.hh"

class G4GenericMessenger;

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Global configuration for setting from the UI under "/config".
 */
class GlobalSetup
{
  public:
    //!@{
    //! \name Type aliases
    using SetupOptions = celeritas::SetupOptions;
    //!@}

  public:
    // Return non-owning pointer to a singleton
    static GlobalSetup* Instance();

    //!@{
    //! \name Demo setup options
    std::string const& GetGeometryFile() const { return geometry_file_; }
    std::string const& GetEventFile() const { return event_file_; }
    int GetRootBufferSize() const { return root_buffer_size_; }
    bool GetWriteSDHits() const { return write_sd_hits_; }
    bool GetSkipMasterSD() const { return skip_master_sd_; }
    bool StripGDMLPointers() const { return strip_gdml_pointers_; }
    //!@}

    //! Get a mutable reference to the setup options for DetectorConstruction
    celeritas::SDSetupOptions& GetSDSetupOptions() { return options_->sd; }

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
    std::shared_ptr<celeritas::SetupOptions> options_;
    std::string geometry_file_;
    std::string event_file_;
    int root_buffer_size_{128000};
    bool write_sd_hits_{false};
    bool skip_master_sd_{false};
    bool strip_gdml_pointers_{true};
    G4ThreeVector field_;

    std::unique_ptr<G4GenericMessenger> messenger_;
};

//---------------------------------------------------------------------------//
}  // namespace demo_geant

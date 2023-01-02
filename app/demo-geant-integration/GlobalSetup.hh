//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/GlobalSetup.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

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
    // Return non-owning pointer to a singleton
    static GlobalSetup* Instance();

    //!@{
    //! Demo setup options
    const std::string& GetGeometryFile() const { return geometry_file_; }
    //!@}

    //! Get a mutable reference to the setup options for DetectorConstruction
    celeritas::SDSetupOptions& GetSDSetupOptions() { return options_->sd; }

    //! Get an immutable reference to the setup options
    std::shared_ptr<const celeritas::SetupOptions> GetSetupOptions() const
    {
        return options_;
    }

  private:
    // Private constructor since we're a singleton
    GlobalSetup();
    ~GlobalSetup();

    // Data
    std::shared_ptr<celeritas::SetupOptions> options_;
    std::string                              geometry_file_;

    std::unique_ptr<G4GenericMessenger> messenger_;
};

//---------------------------------------------------------------------------//
} // namespace demo_geant

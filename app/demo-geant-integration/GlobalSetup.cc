//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/GlobalSetup.cc
//---------------------------------------------------------------------------//
#include "GlobalSetup.hh"

#include <G4GenericMessenger.hh>

#include "corecel/sys/Device.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Return non-owning pointer to a singleton.
 */
GlobalSetup* GlobalSetup::Instance()
{
    static GlobalSetup setup;
    return &setup;
}

//---------------------------------------------------------------------------//
/*!
 * Set configurable properties from the UI.
 */
GlobalSetup::GlobalSetup()
{
    options_   = std::make_shared<celeritas::SetupOptions>();
    messenger_ = std::make_unique<G4GenericMessenger>(
        this, "/setup/", "Demo geant integration setup");

    {
        auto& cmd = messenger_->DeclareProperty("setHepMC3File", hepmc_file_);
        cmd.SetGuidance("Set the HepMC3 file name");
        cmd.SetDefaultValue("");
    }
    {
        auto& cmd = messenger_->DeclareProperty("maxNumTracks",
                                                options_->max_num_tracks);
        cmd.SetGuidance("Set the maximum number of track slots");
        cmd.SetDefaultValue(celeritas::Device::num_devices() > 0 ? "524288"
                                                                 : "64");
    }
    {
        // TODO: expose other options here
    }
}

//---------------------------------------------------------------------------//
//! Default destructor
GlobalSetup::~GlobalSetup() = default;

//---------------------------------------------------------------------------//
} // namespace demo_geant

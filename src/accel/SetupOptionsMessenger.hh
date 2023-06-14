//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SetupOptionsMessenger.hh
//---------------------------------------------------------------------------//
#pragma once
#include <memory>
#include <vector>
#include <G4UIcommand.hh>
#include <G4UIdirectory.hh>
#include <G4UImessenger.hh>

namespace celeritas
{
struct SetupOptions;

//---------------------------------------------------------------------------//
/*!
 * Expose setup options through the Geant4 "macro" UI interface.
 *
 * \warning The given SetupOptions should be global *or* otherwise must exceed
 * the scope of this UI messenger.
 */
class SetupOptionsMessenger : public G4UImessenger
{
  public:
    // Construct with a reference to a setup options instance
    explicit SetupOptionsMessenger(SetupOptions* options);

    // Default destructor
    ~SetupOptionsMessenger();

  protected:
    void SetNewValue(G4UIcommand* command, G4String newValue) override;

  private:
    std::vector<std::unique_ptr<G4UIdirectory>> directories_;
    std::vector<std::unique_ptr<G4UIcommand>> commands_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/SensitiveDetector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <G4VSensitiveDetector.hh>

#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Wrap a local step function as a Geant4 sensitive detector.
 *
 * The given function is called for each completed step inside the sensitive
 * volume, receiving the current worker stream ID and the step data.
 */
class SensitiveDetector : public G4VSensitiveDetector
{
  public:
    //!@{
    //! \name Type aliases
    using FuncLocalStep = std::function<void(StreamId, G4Step const&)>;
    //!@}

  public:
    // Construct from a step callback, or return null if no callback given
    static std::unique_ptr<G4VSensitiveDetector>
    from_hit_function(std::string sd_name, FuncLocalStep f);

    // Construct with a detector name and step callback
    SensitiveDetector(std::string const& name, FuncLocalStep f);

    // Clear hit collection at the beginning of each event
    void Initialize(G4HCofThisEvent*) final;
    // Call the step function for each step in this volume
    G4bool ProcessHits(G4Step* step, G4TouchableHistory*) final;

  private:
    FuncLocalStep hit_func_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

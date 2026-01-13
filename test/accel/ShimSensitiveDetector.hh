//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/ShimSensitiveDetector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <utility>
#include <G4VSensitiveDetector.hh>

#include "corecel/Assert.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Forward hits to a \c std::function.
 */
class ShimSensitiveDetector final : public G4VSensitiveDetector
{
  public:
    //!@{
    //! \name Type aliases
    using HitProcessor = std::function<void(G4Step const*)>;
    //!@}

  public:
    // Construct with name and function
    ShimSensitiveDetector(std::string const& name, HitProcessor&& process_hit)
        : G4VSensitiveDetector(name), process_hit_{std::move(process_hit)}
    {
        CELER_EXPECT(process_hit_);
    }

    void Initialize(G4HCofThisEvent*) final { this->clear(); }
    bool ProcessHits(G4Step* step, G4TouchableHistory*) final
    {
        process_hit_(step);
        return true;  // ignored by Geant4
    }

  private:
    HitProcessor process_hit_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

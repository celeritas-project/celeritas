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
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
namespace test
{
// Forward declaration
StreamId g4_worker_stream();

//---------------------------------------------------------------------------//
/*!
 * Thread-local instance to forward hits to a \c std::function.
 */
class ShimSensitiveDetector final : public G4VSensitiveDetector
{
  public:
    //!@{
    //! \name Type aliases
    using HitProcessor = std::function<void(StreamId, G4Step const&)>;
    //!@}

  public:
    // Construct with name, stream count, and hit function
    ShimSensitiveDetector(std::string const& name,
                          StreamId stream,
                          HitProcessor&& process_hit)
        : G4VSensitiveDetector(name)
        , stream_{stream}
        , process_hit_{std::move(process_hit)}
    {
        CELER_EXPECT(process_hit_);
    }

    void Initialize(G4HCofThisEvent*) final { this->clear(); }
    bool ProcessHits(G4Step* step, G4TouchableHistory*) final
    {
        CELER_EXPECT(stream_);
        process_hit_(stream_, *step);
        return true;  // ignored by Geant4
    }

  private:
    StreamId stream_;
    HitProcessor process_hit_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

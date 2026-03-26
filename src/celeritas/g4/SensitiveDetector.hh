//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/SensitiveDetector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <G4VSensitiveDetector.hh>

#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Wrap a local
 */
class SensitiveDetector : public G4VSensitiveDetector
{
  public:
    //!@{
    //! \name Type aliases
    using LocalStepFunc = std::function<void(StreamId, G4Step const&)>;

    static std::unique_ptr<G4VSensitiveDetector>
    from_hit_function(std::string sd_name, LocalStepFunc f)
    {
        if (!f)
            return nullptr;
        return std::make_unique<SensitiveDetector>(sd_name, std::move(f));
    }

    SensitiveDetector(std::string const& name, LocalStepFunc f)
        : G4VSensitiveDetector(name), hit_func_{std::move(f)}
    {
        CELER_EXPECT(hit_func_);
    }

    void Initialize(G4HCofThisEvent*) final { this->clear(); }
    bool ProcessHits(G4Step* step, G4TouchableHistory*) final
    {
        CELER_EXPECT(step);
        hit_func_(g4_worker_stream(), *step);
        return true;
    }

  private:
    LocalStepFunc hit_func_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

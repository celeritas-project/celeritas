//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/SteppingAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <G4UserSteppingAction.hh>

#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 */
class SteppingAction final : public G4UserSteppingAction
{
  public:
    //!@{
    //! \name Type aliases
    using LocalStepFunc = std::function<void(StreamId, G4Step const&)>;
    //!@}

  public:
    explicit SteppingAction(StreamId sid, LocalStepFunc f)
        : sid_{sid}, callback_{std::move(f)}
    {
        CELER_EXPECT(callback_);
    }

    void UserSteppingAction(G4Step const* step) final
    {
        CELER_EXPECT(step);
        CELER_EXPECT(sid_);
        callback_(sid_, *step);
    }

  private:
    StreamId sid_;
    LocalStepFunc callback_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

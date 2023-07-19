//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/ActionInitialization.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4VUserActionInitialization.hh>

#include "accel/SharedParams.hh"

#include "TrackStepCounter.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Set up demo-specific action initializations.
 */
class ActionInitialization final : public G4VUserActionInitialization
{
  public:
    //!@{
    //! \name Type aliases
    using SPParams = std::shared_ptr<SharedParams>;
    using SPStepCounter = std::shared_ptr<TrackStepCounter>;
    //!@}

  public:
    ActionInitialization();
    void BuildForMaster() const final;
    void Build() const final;

  private:
    SPParams params_;
    SPStepCounter step_counter_;
    mutable bool init_celeritas_;
    mutable bool init_diagnostics_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

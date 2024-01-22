//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/ActionInitialization.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4VUserActionInitialization.hh>

#include "accel/SharedParams.hh"

#include "GeantDiagnostics.hh"

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
    using SPDiagnostics = std::shared_ptr<GeantDiagnostics>;
    //!@}

  public:
    explicit ActionInitialization(SPParams params);
    void BuildForMaster() const final;
    void Build() const final;

    //! Get the number of events to be transported
    int num_events() const { return num_events_; }

  private:
    SPParams params_;
    SPDiagnostics diagnostics_;
    int num_events_{0};
    mutable bool init_shared_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

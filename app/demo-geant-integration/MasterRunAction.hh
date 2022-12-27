//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/MasterRunAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4UserRunAction.hh>

#include "accel/SharedParams.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Set up and tear down Celeritas on the master thread for MT mode.
 */
class MasterRunAction final : public G4UserRunAction
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstOptions = std::shared_ptr<const celeritas::SetupOptions>;
    using SPParams       = std::shared_ptr<celeritas::SharedParams>;
    //!@}

  public:
    MasterRunAction(SPConstOptions options, SPParams params);

    void BeginOfRunAction(const G4Run* run) final;
    void EndOfRunAction(const G4Run* run) final;

  private:
    SPConstOptions options_;
    SPParams       params_;
};

//---------------------------------------------------------------------------//
} // namespace demo_geant

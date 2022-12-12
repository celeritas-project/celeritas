//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/RunAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4UserRunAction.hh>

#include "SetupOptions.hh"
#include "SharedParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up and tear down Celeritas.
 */
class RunAction final : public G4UserRunAction
{
  public:
    //!@{
    //! \name Type aliases
    using SPCOptions = std::shared_ptr<const SetupOptions>;
    //!@}

  public:
    RunAction(SPCOptions options);

    void BeginOfRunAction(const G4Run* run) final;
    void EndOfRunAction(const G4Run* run) final;

  private:
    SPCOptions options_;

    void build_core_params();
};

//---------------------------------------------------------------------------//
} // namespace celeritas

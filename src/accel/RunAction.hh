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
    using SPParams   = std::shared_ptr<SharedParams>;
    //!@}

  public:
    explicit RunAction(SPCOptions options, SPParams params);

    void BeginOfRunAction(const G4Run* run) final;
    void EndOfRunAction(const G4Run* run) final;

  private:
    SPCOptions options_;
    SPParams   params_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

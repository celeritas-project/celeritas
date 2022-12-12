//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/ActionInitialization.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <utility>
#include <G4VUserActionInitialization.hh>

#include "SetupOptions.hh"
#include "SharedParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up Celeritas for EM offloading.
 */
class ActionInitialization final : public G4VUserActionInitialization
{
  public:
    //!@{
    //! \name Type aliases
    using SPCOptions   = std::shared_ptr<const SetupOptions>;
    using SPParams     = std::shared_ptr<SharedParams>;
    using UPUserAction = std::unique_ptr<G4VUserActionInitialization>;
    //!@}

  public:
    ActionInitialization(SPCOptions options, UPUserAction action);
    explicit ActionInitialization(SPCOptions options)
        : ActionInitialization(std::move(options), nullptr)
    {
    }

    void BuildForMaster() const final;
    void Build() const final;

  private:
    SPCOptions   options_;
    SPParams     params_;
    UPUserAction action_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

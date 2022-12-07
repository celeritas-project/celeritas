//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/ActionInitialization.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4VUserActionInitialization.hh>

#include "SetupOptions.hh"

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
    using SPCOptions = std::shared_ptr<const SetupOptions>;
    //!@}

  public:
    explicit ActionInitialization(SPCOptions options);

    void BuildForMaster() const final;
    void Build() const final;

  private:
    SPCOptions options_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

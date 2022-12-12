//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/ActionInitialization.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VUserActionInitialization.hh>

namespace demo_geant
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
    //!@}

  public:
    void BuildForMaster() const final;
    void Build() const final;
};

//---------------------------------------------------------------------------//
} // namespace demo_geant

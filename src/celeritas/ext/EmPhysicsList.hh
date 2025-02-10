//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/EmPhysicsList.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VModularPhysicsList.hh>

#include "GeantPhysicsOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct highly configurable EM-only physics.
 */
class EmPhysicsList : public G4VModularPhysicsList
{
  public:
    //!@{
    //! \name Type aliases
    using Options = GeantPhysicsOptions;
    //!@}

  public:
    // Set up during construction
    explicit EmPhysicsList(Options const& options);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

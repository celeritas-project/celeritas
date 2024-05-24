//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/CelerEmPhysicsList.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VModularPhysicsList.hh>

#include "../GeantPhysicsOptions.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct a user-defined physics list of particles and physics processes.
 */
class CelerEmPhysicsList : public G4VModularPhysicsList
{
  public:
    //!@{
    //! \name Type aliases
    using Options = GeantPhysicsOptions;
    //!@}

  public:
    // Set up during construction
    explicit CelerEmPhysicsList(Options const& options);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

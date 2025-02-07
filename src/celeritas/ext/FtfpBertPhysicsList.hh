//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/FtfpBertPhysicsList.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VModularPhysicsList.hh>

#include "GeantPhysicsOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the FTFP_BERT physics list with configurable EM standard physics.
 */
class FtfpBertPhysicsList : public G4VModularPhysicsList
{
  public:
    //!@{
    //! \name Type aliases
    using Options = GeantPhysicsOptions;
    //!@}

  public:
    // Construct with physics options
    explicit FtfpBertPhysicsList(Options const& options);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

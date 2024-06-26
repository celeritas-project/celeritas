//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/CelerFTFPBert.hh
// TODO: Move out of detail since this is used by celer-g4
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
 * Construct the FTFP_BERT physics list with modified EM standard physics.
 */
class CelerFTFPBert : public G4VModularPhysicsList
{
  public:
    //!@{
    //! \name Type aliases
    using Options = GeantPhysicsOptions;
    //!@}

  public:
    // Construct with physics options
    explicit CelerFTFPBert(Options const& options);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

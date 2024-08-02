//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/CelerOpticalPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VPhysicsConstructor.hh>

#include "../GeantOpticalPhysicsOptions.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas-supported Optical physics.
 */
class CelerOpticalPhysics : public G4VPhysicsConstructor
{
  public:
    //!@{
    //! \name Type aliases
    using Options = GeantOpticalPhysicsOptions;
    //!@}

  public:
    // Set up during construction
    explicit CelerOpticalPhysics(Options const& options);

    // Set up minimal EM particle list
    void ConstructParticle() override;
    // Set up process list
    void ConstructProcess() override;

  private:
    Options options_;

    // Return true if process is activated
    bool process_is_active(std::string const& process);
};
}  // namespace detail
}  // namespace celeritas

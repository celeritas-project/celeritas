//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/PrimaryGeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VUserPrimaryGeneratorAction.hh>

namespace celeritas
{
namespace example
{
//---------------------------------------------------------------------------//
/*!
 * Generate primaries.
 */
class PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    // Construct empty
    PrimaryGeneratorAction();

    // Place primaries in the event simulation
    void GeneratePrimaries(G4Event* event) final;
};

}  // namespace example
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/PGPrimaryGeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <random>
#include <G4Event.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleGun.hh>
#include <G4VUserPrimaryGeneratorAction.hh>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Generate events from a particle gun.
 */
class PGPrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    //!@{
    //! \name Type aliases
    using Real3 = Array<real_type, 3>;
    using EnergySampler = std::function<real_type(std::mt19937&)>;
    using PositionSampler = std::function<Real3(std::mt19937&)>;
    using DirectionSampler = std::function<Real3(std::mt19937&)>;
    //!@}

  public:
    // Construct primary action
    PGPrimaryGeneratorAction();

    // Generate events
    void GeneratePrimaries(G4Event* event) final;

  private:
    G4ParticleGun gun_;
    std::mt19937 rng_;
    size_type num_events_{};
    size_type primaries_per_event_{};
    std::vector<G4ParticleDefinition*> particle_def_;
    EnergySampler sample_energy_;
    PositionSampler sample_pos_;
    DirectionSampler sample_dir_;
    size_type primary_count_{0};
    size_type event_count_{0};
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

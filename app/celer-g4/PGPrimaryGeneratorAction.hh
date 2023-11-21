//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/PGPrimaryGeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <G4Event.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleGun.hh>
#include <G4VUserPrimaryGeneratorAction.hh>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "orange/Types.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Generate events from a particle gun.
 *
 * This generates primary particles with energy, position, and direction
 * sampled from distributions specified by the user in \c
 * PrimaryGeneratorOptions.
 *
 * \sa PrimaryGenerator
 */
class PGPrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    //!@{
    //! \name Type aliases
    using EnergySampler = std::function<real_type(PrimaryGeneratorEngine&)>;
    using PositionSampler = std::function<Real3(PrimaryGeneratorEngine&)>;
    using DirectionSampler = std::function<Real3(PrimaryGeneratorEngine&)>;
    //!@}

  public:
    // Construct primary action
    explicit PGPrimaryGeneratorAction(PrimaryGeneratorOptions const& opts);

    // Generate events
    void GeneratePrimaries(G4Event* event) final;

  private:
    G4ParticleGun gun_;
    PrimaryGeneratorEngine rng_;
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

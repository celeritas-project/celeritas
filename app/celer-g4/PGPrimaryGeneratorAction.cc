//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/PGPrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "PGPrimaryGeneratorAction.hh"

#include <random>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4ParticleTable.hh>

#include "corecel/Macros.hh"
#include "celeritas/ext/Convert.geant.hh"
#include "celeritas/ext/GeantUtils.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct primary action.
 */
PGPrimaryGeneratorAction::PGPrimaryGeneratorAction(
    PrimaryGeneratorOptions const& options)
{
    CELER_EXPECT(options);

    // Generate one particle at each call to \c GeneratePrimaryVertex()
    gun_.SetNumberOfParticles(1);

    seed_ = options.seed;
    num_events_ = options.num_events;
    primaries_per_event_ = options.primaries_per_event;
    sample_energy_ = make_energy_sampler(options.energy);
    sample_pos_ = make_position_sampler(options.position);
    sample_dir_ = make_direction_sampler(options.direction);

    // Set the particle definitions
    particle_def_.reserve(options.pdg.size());
    for (auto const& pdg : options.pdg)
    {
        particle_def_.push_back(
            G4ParticleTable::GetParticleTable()->FindParticle(pdg.get()));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Generate primaries from a particle gun.
 */
void PGPrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    CELER_EXPECT(event);
    CELER_EXPECT(event->GetEventID() >= 0);

    size_type event_id = event->GetEventID();
    if (event_id >= num_events_)
    {
        return;
    }

    // Seed with an independent value for each event. Since Geant4 schedules
    // events dynamically, the same event ID may not be mapped to the same
    // thread across multiple runs. For reproducibility, Geant4 reseeds each
    // worker thread's RNG at the start of each event with a seed calculated
    // from the event ID.
    rng_.seed(seed_ + event_id);

    for (size_type i = 0; i < primaries_per_event_; ++i)
    {
        gun_.SetParticleDefinition(particle_def_[i % particle_def_.size()]);
        gun_.SetParticlePosition(
            convert_to_geant(sample_pos_(rng_), clhep_length));
        gun_.SetParticleMomentumDirection(
            convert_to_geant(sample_dir_(rng_), 1));
        gun_.SetParticleEnergy(
            convert_to_geant(sample_energy_(rng_), CLHEP::MeV));
        gun_.GeneratePrimaryVertex(event);

        if (CELERITAS_DEBUG)
        {
            CELER_ASSERT(G4VPrimaryGenerator::CheckVertexInsideWorld(
                gun_.GetParticlePosition()));
        }
    }

    CELER_ENSURE(event->GetNumberOfPrimaryVertex()
                 == static_cast<int>(primaries_per_event_));
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

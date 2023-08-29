//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/PGPrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "PGPrimaryGeneratorAction.hh"

#include <G4ParticleTable.hh>

#include "corecel/Macros.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"
#include "celeritas/random/distribution/DeltaDistribution.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"
#include "celeritas/random/distribution/UniformBoxDistribution.hh"

#include "GlobalSetup.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct primary action.
 */
PGPrimaryGeneratorAction::PGPrimaryGeneratorAction()
{
    if (G4Threading::IsMultithreadedApplication())
    {
        rng_.seed(G4Threading::G4GetThreadId());
    }

    // Generate one particle at each call to \c GeneratePrimaryVertex()
    gun_.SetNumberOfParticles(1);

    auto options = GlobalSetup::Instance()->GetPrimaryGeneratorOptions();
    CELER_ASSERT(options);

    num_events_ = options.num_events;
    primaries_per_event_ = options.primaries_per_event;

    // Set the particle definitions
    particle_def_.reserve(options.pdg.size());
    for (auto const& pdg : options.pdg)
    {
        particle_def_.push_back(
            G4ParticleTable::GetParticleTable()->FindParticle(pdg.get()));
    }

    using DS = DistributionSelection;
    {
        // Create energy distribution
        auto const& p = options.energy.params;
        switch (options.energy.distribution)
        {
            case DS::delta:
                CELER_ASSERT(p.size() == 1);
                sample_energy_ = DeltaDistribution<real_type>(p[0]);
                break;
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }
    {
        // Create spatial distribution
        auto const& p = options.position.params;
        switch (options.position.distribution)
        {
            case DS::delta:
                CELER_ASSERT(p.size() == 3);
                sample_pos_ = DeltaDistribution<Real3>(Real3{p[0], p[1], p[2]});
                break;
            case DS::box:
                CELER_ASSERT(p.size() == 6);
                sample_pos_ = UniformBoxDistribution<real_type>(
                    Real3{p[0], p[1], p[2]}, Real3{p[3], p[4], p[5]});
                break;
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }
    {
        // Create angular distribution
        auto const& p = options.direction.params;
        switch (options.direction.distribution)
        {
            case DS::delta:
                CELER_ASSERT(p.size() == 3);
                sample_dir_ = DeltaDistribution<Real3>(Real3{p[0], p[1], p[2]});
                break;
            case DS::isotropic:
                CELER_ASSERT(p.empty());
                sample_dir_ = IsotropicDistribution<real_type>();
                break;
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Generate primaries from a particle gun.
 */
void PGPrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    CELER_EXPECT(event);

    if (event_count_ == num_events_)
    {
        return;
    }

    for (size_type i = 0; i < primaries_per_event_; ++i)
    {
        gun_.SetParticleDefinition(
            particle_def_[primary_count_ % particle_def_.size()]);
        gun_.SetParticlePosition(
            convert_to_geant(sample_pos_(rng_), CLHEP::cm));
        gun_.SetParticleMomentumDirection(
            convert_to_geant(sample_dir_(rng_), 1));
        gun_.SetParticleEnergy(
            convert_to_geant(sample_energy_(rng_), CLHEP::MeV));
        gun_.GeneratePrimaryVertex(event);
        ++primary_count_;

        if (CELERITAS_DEBUG)
        {
            CELER_ASSERT(G4VPrimaryGenerator::CheckVertexInsideWorld(
                gun_.GetParticlePosition()));
        }
    }
    ++event_count_;

    CELER_ENSURE(event->GetNumberOfPrimaryVertex()
                 == static_cast<int>(primaries_per_event_));
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

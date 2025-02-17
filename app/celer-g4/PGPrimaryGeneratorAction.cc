//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/PGPrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "PGPrimaryGeneratorAction.hh"

#include <random>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4ParticleTable.hh>

#include "corecel/Macros.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantUnits.hh"
#include "celeritas/inp/Events.hh"
#include "celeritas/phys/Primary.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
std::vector<ParticleId> make_particle_ids(std::vector<PDGNumber> const& all_pdg)
{
    CELER_VALIDATE(!all_pdg.empty(),
                   << "primary generator has no input particles");

    auto* par_table = G4ParticleTable::GetParticleTable();
    CELER_ASSERT(par_table);

    // Find and Geant4 particles and create fake Particle IDs
    std::vector<ParticleId> result(all_pdg.size());
    for (auto i : range(all_pdg.size()))
    {
        auto const& pdg = all_pdg[i];
        CELER_EXPECT(pdg);
        auto* p = par_table->FindParticle(pdg.get());
        CELER_VALIDATE(
            p, << "particle with PDG " << pdg.get() << " is not loaded");
        result[i] = ParticleId(i);
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct primary action.
 */
PGPrimaryGeneratorAction::PGPrimaryGeneratorAction(Input const& i)
    : pdg_(i.pdg), generate_primaries_(i, make_particle_ids(i.pdg))
{
    // Generate one particle at each call to \c GeneratePrimaryVertex()
    gun_.SetNumberOfParticles(1);
}

//---------------------------------------------------------------------------//
/*!
 * Generate primaries from a particle gun.
 */
void PGPrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    CELER_EXPECT(event);
    CELER_EXPECT(event->GetEventID() >= 0);

    // Seed with an independent value for each event. Since Geant4 schedules
    // events dynamically, the same event ID may not be mapped to the same
    // thread across multiple runs. For reproducibility, Geant4 reseeds each
    // worker thread's RNG at the start of each event with a seed calculated
    // from the event ID.
    generate_primaries_.seed(id_cast<UniqueEventId>(event->GetEventID()));

    auto primaries = generate_primaries_();

    for (auto const& p : primaries)
    {
        CELER_ASSERT(p.particle_id < pdg_.size());
        gun_.SetParticleDefinition(
            G4ParticleTable::GetParticleTable()->FindParticle(
                pdg_[p.particle_id.unchecked_get()].get()));
        gun_.SetParticlePosition(convert_to_geant(p.position, clhep_length));
        gun_.SetParticleMomentumDirection(convert_to_geant(p.direction, 1));
        gun_.SetParticleEnergy(convert_to_geant(p.energy, CLHEP::MeV));
        gun_.GeneratePrimaryVertex(event);

        if (CELERITAS_DEBUG)
        {
            CELER_ASSERT(G4VPrimaryGenerator::CheckVertexInsideWorld(
                gun_.GetParticlePosition()));
        }
    }

    CELER_ENSURE(event->GetNumberOfPrimaryVertex()
                 == static_cast<int>(primaries.size()));
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

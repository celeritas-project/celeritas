//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleParamsLoader.cc
//---------------------------------------------------------------------------//
#include "ParticleParamsLoader.hh"

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>

#include "base/Assert.hh"
#include "ImportParticle.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with RootLoader.
 */
ParticleParamsLoader::ParticleParamsLoader(RootLoader root_loader)
    : root_loader_(root_loader)
{
    CELER_ENSURE(root_loader);
}

//---------------------------------------------------------------------------//
/*!
 * Load ParticleParams data.
 */
const std::shared_ptr<const ParticleParams> ParticleParamsLoader::operator()()
{
    // Open the 'particles' branch and reserve size for the converted data
    const auto             tfile = root_loader_.get();
    std::unique_ptr<TTree> tree_particles(tfile->Get<TTree>("particles"));
    CELER_ASSERT(tree_particles);

    ParticleParams::Input defs(tree_particles->GetEntries());
    CELER_ASSERT(!defs.empty());

    // Load the particle data
    ImportParticle  particle;
    ImportParticle* temp_particle_ptr = &particle;

    int err_code = tree_particles->SetBranchAddress("ImportParticle",
                                                    &temp_particle_ptr);
    CELER_ASSERT(err_code >= 0);

    for (auto i : range(defs.size()))
    {
        // Load a single entry into particle
        particle.name.clear();
        tree_particles->GetEntry(i);
        CELER_ASSERT(!particle.name.empty());

        // Convert metadata
        defs[i].name     = particle.name;
        defs[i].pdg_code = PDGNumber{particle.pdg};
        CELER_ASSERT(defs[i].pdg_code);

        // Convert data
        defs[i].mass           = units::MevMass{particle.mass};
        defs[i].charge         = units::ElementaryCharge{particle.charge};
        defs[i].decay_constant = (particle.is_stable
                                      ? ParticleDef::stable_decay_constant()
                                      : 1. / particle.lifetime);
    }

    // Sort by increasing mass, then by PDG code (positive before negative of
    // the same absolute value). Placing lighter particles
    // (more likely to be created by various processes, so more "light
    // particle" tracks) together at the beginning of the list will make it
    // easier to human-read the particles while debugging, and having them
    // at adjacent memory locations could improve cacheing.
    auto to_particle_key = [](const auto& inp) {
        int pdg = inp.pdg_code.get();
        return std::make_tuple(inp.mass, std::abs(pdg), pdg < 0);
    };
    std::sort(defs.begin(),
              defs.end(),
              [to_particle_key](const auto& lhs, const auto& rhs) {
                  return to_particle_key(lhs) < to_particle_key(rhs);
              });

    // Construct ParticleParams from the definitions
    return std::make_shared<ParticleParams>(std::move(defs));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
